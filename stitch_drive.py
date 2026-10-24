#!/usr/bin/env python3
"""StabStitch++ on the driving videos.

The released multi-video example (Full_model_inference/Codes/test_online_tra_threeview.py)
stitches three overlapping videos: video1 overlaps video2, and video2 overlaps
video3. On the driving rig that shared view is the back camera, so the triple
is cam4 | cam5 | cam3.

A scene is one video. Spatial, temporal, and smooth warps run online across
the frame sequence (smooth window = 7, the same buffer as the official
script). The finished panorama is then pulled back onto the original camera-5
plane by the inverse of camera 5's final mesh, so it lines up with the sparse
GT. Each frame is written as:

  outputs/<dataset>/<scene>/threeview/stitch/<frame>.jpg
      the complete restored panorama, white outside the content
  outputs/metric_eval/<dataset>/threeview/<scene>/<frame>.png
      camera 5's box, plus one camera-5 width on the left and one on the
      right, resized to the sparse GT

Examples:
  python stitch_drive.py --dataset nuscenes --scene 002 --max-frames 8 --gpu 0
  python stitch_drive.py --dataset nuscenes --scene 002 --frame 19 --gpu 0
  python stitch_drive.py --dataset lyft1920 --all --gpu 0
"""
import argparse
import os
import re
import subprocess
import sys

# CUDA_VISIBLE_DEVICES is read at torch import.
_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--gpu", default="0")
_PRE_ARGS, _ = _PRE.parse_known_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(_PRE_ARGS.gpu)

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import GaussianBlur

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Full_model_inference" / "Codes"
sys.path.insert(0, str(CODE_DIR))

import grid_res  # noqa: E402
import utils.torch_tps_transform as torch_tps_transform  # noqa: E402
import utils.torch_tps_transform_point as torch_tps_transform_point  # noqa: E402
from smooth_network import SmoothNet, build_SmoothNet  # noqa: E402
from spatial_network import SpatialNet, build_SpatialNet, get_norm_mesh, get_rigid_mesh  # noqa: E402
from temporal_network import TemporalNet, build_TemporalNet  # noqa: E402

WORK = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub")
DATASET_REL = {
    "nuscenes": Path("nuscenes/processed_10Hz_v2"),
    "lyft1920": Path("lyft/lyft_val1920_3cams"),
    "lyft1224": Path("lyft/lyft_val1224_3cams"),
    "widedrive3": Path("WideDrive3_processed/WideDriveVal"),
}
GT_SPECS = {
    "nuscenes": {
        "root": Path("nuscenes/sparseWideFOVImages3_1554x294"),
        "size": (1554, 294),
        "suffix": "_5_sparse_wide.png",
        "val_list": Path("nuscenes/processed_10Hz_v2/nuScenes_Val.txt"),
    },
    "lyft1920": {
        "root": Path("lyft/1920_sparseWideFOVImages3_1554x294"),
        "size": (1554, 294),
        "suffix": "_5_sparse_wide.png",
        "val_list": Path("lyft/lyft_val1920_3cams/lyft_val1920.txt"),
    },
    "lyft1224": {
        "root": Path("lyft/1224_sparseWideFOVImages3_1554x434"),
        "size": (1554, 434),
        "suffix": "_5_sparse_wide.png",
        "val_list": Path("lyft/lyft_val1224_3cams/lyft_val1224.txt"),
    },
    "widedrive3": {
        "root": Path("WideDrive_processed/sparseWideFOVImages3_1554x294"),
        "size": (1554, 294),
        "suffix": "_2_sparse_wide.png",
        "val_list": Path("WideDrive3_processed/WideDriveVal/val.txt"),
    },
}
WEIGHTS = {
    "spatial": ROOT / "pretrained" / "spatial_warp.pth",
    "temporal": ROOT / "pretrained" / "temporal_warp.pth",
    "smooth": ROOT / "pretrained" / "smooth_warp.pth",
}
NET_H = 360
NET_W = 480
GRID_H = grid_res.GRID_H
GRID_W = grid_res.GRID_W
# cam4 overlaps cam5, cam5 overlaps cam3. cam5 is the shared middle view.
CAM_LEFT = 4
CAM_MID = 5
CAM_RIGHT = 3
SMOOTH_WINDOW = 7


def resolve_data(relative):
    local = ROOT / "datasets" / relative
    if local.exists():
        return local
    shared = WORK / "datasets" / relative
    if shared.exists():
        return shared
    raise SystemExit(f"dataset path not found: {local} or {shared}")


def parse_args():
    parser = argparse.ArgumentParser(description="StabStitch++ 3-camera driving video stitch")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REL))
    parser.add_argument("--scene", default=None)
    parser.add_argument("--all", action="store_true", help="every scene in the dataset val list")
    parser.add_argument("--frame", action="append", default=None, help="frame id to save; still uses video context")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-height", type=int, default=480, help="warp resolution; the network itself always sees 360x480")
    parser.add_argument("--max-frames", type=int, default=0, help="only the first N video frames (0 = all)")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--metric-dir", default=str(ROOT / "outputs" / "metric_eval"))
    parser.add_argument("--warp-mode", default="NORMAL", choices=["NORMAL", "FAST"])
    parser.add_argument("--gt-width", type=int, default=None)
    parser.add_argument("--gt-height", type=int, default=None)
    parser.add_argument("--max-canvas", type=int, default=8000, help="skip a scene whose stitch canvas exceeds this side")
    return parser.parse_args()


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def scene_dir(dataset, scene):
    root = resolve_data(DATASET_REL[dataset])
    name = str(scene)
    candidates = [name]
    if name.isdigit():
        candidates.append(str(int(name)).zfill(3))
        candidates.append(name.zfill(3))
    seen = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.append(cand)
        path = root / cand
        if path.is_dir():
            return path
    raise SystemExit(f"scene not found under {root}: tried {seen}")


def frame_token(frame):
    text = str(frame)
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def list_video_frames(scene):
    """Frames that have cam4, cam5, and cam3, in filename order."""
    image_dir = scene / "images"
    frames = []
    if not image_dir.is_dir():
        return frames
    for name in sorted(os.listdir(image_dir)):
        if not name.endswith(f"_{CAM_MID}.jpg"):
            continue
        frame = name[: -len(f"_{CAM_MID}.jpg")]
        paths = [image_dir / f"{frame}_{cam}.jpg" for cam in (CAM_LEFT, CAM_MID, CAM_RIGHT)]
        if all(path.is_file() for path in paths):
            frames.append(frame)
    return frames


def has_gt(gt_dir, frame, suffix):
    return (gt_dir / "rgb" / f"{frame}{suffix}").is_file()


def load_bgr(path, size_wh):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    if (image.shape[1], image.shape[0]) != size_wh:
        image = cv2.resize(image, size_wh, interpolation=cv2.INTER_AREA)
    return image


def net_tensor(image):
    small = cv2.resize(image, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    array = small.astype(np.float32).transpose(2, 0, 1)
    array = (array / 127.5) - 1.0
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def bgr_tensor(image):
    array = image.astype(np.float32).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def recover_mesh(norm_mesh, height, width):
    batch_size = norm_mesh.size()[0]
    mesh_w = (norm_mesh[..., 0] + 1) * float(width) / 2.0
    mesh_h = (norm_mesh[..., 1] + 1) * float(height) / 2.0
    mesh = torch.stack([mesh_w, mesh_h], 2)
    return mesh.reshape([batch_size, GRID_H + 1, GRID_W + 1, 2])


def linear_blender(ref, tgt, ref_m, tgt_m):
    """Official seam blend from test_online_tra_threeview.py."""
    blur = GaussianBlur(kernel_size=(21, 21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)
    if r1.numel() == 0 or r2.numel() == 0:
        raise RuntimeError("a warped view is empty")
    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())
    vec = (center2[0] - center1[0], center2[1] - center1[1])
    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_only = ref_m[:, 0].unsqueeze(1) - ovl
    rows, cols = torch.nonzero(ovl[0, 0], as_tuple=True)
    ovl_mask = torch.zeros_like(ref_only)
    if rows.numel() > 0:
        proj = (rows - center1[0]) * vec[0] + (cols - center1[1]) * vec[1]
        ovl_mask[ovl.bool()] = (proj - proj.min()) / (proj.max() - proj.min() + 1e-3)
    mask1 = (blur(ref_only + (1 - ovl_mask) * ref_m[:, 0].unsqueeze(1)) * ref_m + ref_only).clamp(0, 1)
    mask2 = (1 - mask1) * tgt_m
    return ref * mask1 + tgt * mask2


def load_nets():
    if not torch.cuda.is_available():
        raise SystemExit("StabStitch++ inference uses CUDA")
    missing = [name for name, path in WEIGHTS.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing pretrained weights: {missing} under {ROOT / 'pretrained'}")
    spatial_net = SpatialNet().cuda().eval()
    temporal_net = TemporalNet().cuda().eval()
    smooth_net = SmoothNet().cuda().eval()
    for net, key in ((spatial_net, "spatial"), (temporal_net, "temporal"), (smooth_net, "smooth")):
        checkpoint = torch.load(WEIGHTS[key], map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        net.load_state_dict(state)
        print(f"loaded {WEIGHTS[key].name}")
    return spatial_net, temporal_net, smooth_net


def scale_mesh(mesh, img_h, img_w):
    return torch.stack([mesh[..., 0] * img_w / float(NET_W), mesh[..., 1] * img_h / float(NET_H)], 4)


def canvas_limits(meshes):
    width_max = torch.max(torch.stack([mesh[..., 0].max() for mesh in meshes]))
    width_min = torch.min(torch.stack([mesh[..., 0].min() for mesh in meshes]))
    height_max = torch.max(torch.stack([mesh[..., 1].max() for mesh in meshes]))
    height_min = torch.min(torch.stack([mesh[..., 1].min() for mesh in meshes]))
    return width_min, width_max, height_min, height_max


def shift_mesh(mesh, width_min, height_min):
    return torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 4)


def pair_motions(spatial_net, temporal_net, smooth_net, left_tensors, right_tensors):
    """Online spatial + temporal + smooth warps for one overlapping pair."""
    smotion1 = []
    smotion2 = []
    with torch.no_grad():
        for left, right in zip(left_tensors, right_tensors):
            spatial_out = build_SpatialNet(spatial_net, left.cuda(), right.cuda())
            smotion1.append(spatial_out["motion1"])
            smotion2.append(spatial_out["motion2"])
        temporal1 = build_TemporalNet(temporal_net, left_tensors)["motion_list"]
        temporal2 = build_TemporalNet(temporal_net, right_tensors)["motion_list"]

    rigid = get_rigid_mesh(1, NET_H, NET_W)
    norm_rigid = get_norm_mesh(rigid, NET_H, NET_W)
    smesh1, smesh2, tsmotion1, tsmotion2 = [], [], [], []
    for index, (motion_a, motion_b) in enumerate(zip(smotion1, smotion2)):
        mesh_a = rigid + motion_a
        mesh_b = rigid + motion_b
        if index == 0:
            delta_a = motion_a * 0
            delta_b = motion_b * 0
        else:
            delta_a = _temporal_delta(temporal1[index], smotion1[index - 1], motion_a, rigid, norm_rigid)
            delta_b = _temporal_delta(temporal2[index], smotion2[index - 1], motion_b, rigid, norm_rigid)
        smesh1.append(mesh_a)
        smesh2.append(mesh_b)
        tsmotion1.append(delta_a)
        tsmotion2.append(delta_b)

    count = len(tsmotion1)
    if count < SMOOTH_WINDOW:
        raise RuntimeError(f"smooth warp needs >= {SMOOTH_WINDOW} frames, got {count}")
    ori1 = ori2 = smooth1 = smooth2 = None
    with torch.no_grad():
        for start in range(count - SMOOTH_WINDOW + 1):
            window1 = tsmotion1[start:start + SMOOTH_WINDOW]
            window2 = tsmotion2[start:start + SMOOTH_WINDOW]
            window1[0] = window1[0] * 0
            window2[0] = window2[0] * 0
            smooth_out = build_SmoothNet(
                smooth_net,
                window1,
                window2,
                smesh1[start:start + SMOOTH_WINDOW],
                smesh2[start:start + SMOOTH_WINDOW],
            )
            if start == 0:
                ori1 = smooth_out["ori_mesh1"]
                smooth1 = smooth_out["smooth_mesh1"]
                ori2 = smooth_out["ori_mesh2"]
                smooth2 = smooth_out["smooth_mesh2"]
            else:
                ori1 = torch.cat((ori1, smooth_out["ori_mesh1"][:, -1:].contiguous()), 1)
                smooth1 = torch.cat((smooth1, smooth_out["smooth_mesh1"][:, -1:].contiguous()), 1)
                ori2 = torch.cat((ori2, smooth_out["ori_mesh2"][:, -1:].contiguous()), 1)
                smooth2 = torch.cat((smooth2, smooth_out["smooth_mesh2"][:, -1:].contiguous()), 1)
    del ori1, ori2
    return smooth1, smooth2


def _temporal_delta(tmotion, prev_smotion, smotion, rigid, norm_rigid):
    prev_smesh = rigid + prev_smotion
    tmesh = rigid + tmotion
    tsmesh = torch_tps_transform_point.transformer(
        get_norm_mesh(tmesh, NET_H, NET_W),
        norm_rigid,
        get_norm_mesh(prev_smesh, NET_H, NET_W),
    )
    return recover_mesh(tsmesh, NET_H, NET_W) - (rigid + smotion)


def align_three(mesh_left_on_mid, mesh_mid_from_left, mesh_mid_from_right, mesh_right_on_mid):
    """Official three-view mesh alignment: the two estimates of cam5 share one plane."""
    offset = (mesh_mid_from_left - mesh_mid_from_right).reshape(
        mesh_mid_from_left.shape[0], mesh_mid_from_left.shape[1], -1, 2
    )
    offset = torch.mean(offset, 2).unsqueeze(2).unsqueeze(2)
    mesh_mid_from_right = mesh_mid_from_right + offset
    mesh_right_on_mid = mesh_right_on_mid + offset
    middle = (mesh_mid_from_left + mesh_mid_from_right) / 2.0

    width_min, width_max, height_min, height_max = canvas_limits(
        [mesh_left_on_mid, mesh_mid_from_left, mesh_mid_from_right, mesh_right_on_mid]
    )
    out_h = int((height_max - height_min).item())
    out_w = int((width_max - width_min).item())
    mesh_left_on_mid = shift_mesh(mesh_left_on_mid, width_min, height_min)
    mesh_mid_from_left = shift_mesh(mesh_mid_from_left, width_min, height_min)
    mesh_mid_from_right = shift_mesh(mesh_mid_from_right, width_min, height_min)
    mesh_right_on_mid = shift_mesh(mesh_right_on_mid, width_min, height_min)
    middle = shift_mesh(middle, width_min, height_min)

    left_frames = []
    right_frames = []
    for index in range(middle.shape[1]):
        norm_left = get_norm_mesh(mesh_left_on_mid[:, index], out_h, out_w)
        norm_mid_l = get_norm_mesh(mesh_mid_from_left[:, index], out_h, out_w)
        norm_mid_r = get_norm_mesh(mesh_mid_from_right[:, index], out_h, out_w)
        norm_right = get_norm_mesh(mesh_right_on_mid[:, index], out_h, out_w)
        norm_middle = get_norm_mesh(middle[:, index], out_h, out_w)
        left = torch_tps_transform_point.transformer(norm_left, norm_mid_l, norm_middle)
        right = torch_tps_transform_point.transformer(norm_right, norm_mid_r, norm_middle)
        left_frames.append(recover_mesh(left, out_h, out_w))
        right_frames.append(recover_mesh(right, out_h, out_w))
    mesh_left = torch.stack(left_frames, 1)
    mesh_right = torch.stack(right_frames, 1)

    width_min, width_max, height_min, height_max = canvas_limits([mesh_left, middle, mesh_right])
    out_h = int((height_max - height_min).item())
    out_w = int((width_max - width_min).item())
    if out_h < 2 or out_w < 2:
        raise RuntimeError(f"empty canvas {out_w}x{out_h}")
    return mesh_left, middle, mesh_right, width_min, height_min, out_h, out_w


def inverse_to_cam5(color, valid, cam5, mesh, width_min, height_min, img_h, img_w, warp_mode, max_canvas):
    """Pull one stitched canvas back onto the original camera-5 plane.

    ``mesh`` is camera 5's final mesh in the same coordinates as the forward
    warp. Swapping its control points samples the panorama at those mesh
    locations and writes them back onto camera 5's rigid grid. Content outside
    that grid is the same thin-plate extrapolation, so the side cameras come
    back with it.
    """
    stitch_h = int(color.shape[2])
    stitch_w = int(color.shape[3])
    moved = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    rigid = get_rigid_mesh(1, img_h, img_w)
    norm_mesh = get_norm_mesh(moved, stitch_h, stitch_w)
    norm_rigid = get_norm_mesh(rigid, img_h, img_w)

    valid_np = (valid[0, 0] > 0.5).detach().cpu().numpy()
    pts = _content_outline(valid_np)
    if len(pts) < 10:
        raise RuntimeError("inverse warp: stitched image is empty")
    pts = torch.from_numpy(pts).unsqueeze(0).to(color.device)
    pts_n = pts.clone()
    pts_n[..., 0] = pts[..., 0] * 2.0 / float(stitch_w) - 1.0
    pts_n[..., 1] = pts[..., 1] * 2.0 / float(stitch_h) - 1.0
    mapped = torch_tps_transform_point.transformer(pts_n, norm_mesh, norm_rigid)
    mx = (mapped[0, :, 0] + 1.0) * float(img_w) / 2.0
    my = (mapped[0, :, 1] + 1.0) * float(img_h) / 2.0
    # Drop only non-finite thin-plate spikes. The outline itself is kept, so the
    # side cameras are not trimmed the way a percentile canvas trims them.
    keep = torch.isfinite(mx) & torch.isfinite(my)
    keep = keep & (mx > -6.0 * img_w) & (mx < 7.0 * img_w)
    keep = keep & (my > -6.0 * img_h) & (my < 7.0 * img_h)
    mx, my = mx[keep], my[keep]
    if mx.numel() < 10:
        raise RuntimeError("inverse warp: camera-5 mapping left no usable pixels")
    margin = 32.0
    x0 = min(float(mx.min()), 0.0) - margin
    y0 = min(float(my.min()), 0.0) - margin
    x1 = max(float(mx.max()), float(img_w)) + margin
    y1 = max(float(my.max()), float(img_h)) + margin
    out_w = int(np.ceil(x1 - x0))
    out_h = int(np.ceil(y1 - y0))
    if out_w < 2 or out_h < 2:
        raise RuntimeError(f"inverse canvas empty {out_w}x{out_h}")
    if max(out_w, out_h) > max_canvas:
        raise RuntimeError(f"inverse canvas {out_w}x{out_h} exceeds --max-canvas {max_canvas}")

    src = torch.stack(
        [
            (rigid[..., 0] - x0) * 2.0 / float(out_w) - 1.0,
            (rigid[..., 1] - y0) * 2.0 / float(out_h) - 1.0,
        ],
        3,
    ).reshape(1, -1, 2)
    packed = torch.cat([color, valid, cam5], 1)
    warped = torch_tps_transform.transformer(packed, src, norm_mesh, (out_h, out_w), mode=warp_mode)
    return warped[:, 0:3], warped[:, 3:4], warped[:, 4:5]


def _content_outline(valid):
    """Stitch-plane points whose inverse image is the full content extent."""
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    step = max(1, int(np.sqrt(len(xs) / 12000.0)))
    grid = np.stack([xs[::step], ys[::step]], 1)
    contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return grid.astype(np.float32)
    border = np.concatenate([contour.reshape(-1, 2) for contour in contours], 0)
    if len(border) > 20000:
        border = border[:: max(1, len(border) // 20000)]
    return np.concatenate([grid, border], 0).astype(np.float32)


def warp_frame(image, mesh, width_min, height_min, out_h, out_w, warp_mode):
    moved = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    norm_mesh = get_norm_mesh(moved, out_h, out_w)
    rigid = get_rigid_mesh(1, image.shape[2], image.shape[3])
    norm_rigid = get_norm_mesh(rigid, image.shape[2], image.shape[3])
    mask = torch.ones_like(image[:, :1])
    packed = torch.cat([image, mask], 1).cuda()
    warped = torch_tps_transform.transformer(packed, norm_mesh, norm_rigid, (out_h, out_w), mode=warp_mode)
    color = warped[:, 0:3]
    valid = warped[:, 3:4]
    return color, valid


def stitch_clip(spatial_net, temporal_net, smooth_net, frame_paths, size_wh, warp_mode, max_canvas):
    """Return per-frame BGR uint8, valid mask, and cam5 mask on the original cam5 plane."""
    left_tensors, mid_tensors, right_tensors = [], [], []
    for path_left, path_mid, path_right in frame_paths:
        left_tensors.append(net_tensor(load_bgr(path_left, size_wh)))
        mid_tensors.append(net_tensor(load_bgr(path_mid, size_wh)))
        right_tensors.append(net_tensor(load_bgr(path_right, size_wh)))

    # Pair 1: cam4 (video1) with cam5 (video2). Pair 2: cam5 (video1) with cam3 (video2).
    left_mesh, mid_from_left = pair_motions(spatial_net, temporal_net, smooth_net, left_tensors, mid_tensors)
    mid_from_right, right_mesh = pair_motions(spatial_net, temporal_net, smooth_net, mid_tensors, right_tensors)
    img_h, img_w = size_wh[1], size_wh[0]
    left_mesh = scale_mesh(left_mesh, img_h, img_w)
    mid_from_left = scale_mesh(mid_from_left, img_h, img_w)
    mid_from_right = scale_mesh(mid_from_right, img_h, img_w)
    right_mesh = scale_mesh(right_mesh, img_h, img_w)
    left_mesh, middle, right_mesh, width_min, height_min, out_h, out_w = align_three(
        left_mesh, mid_from_left, mid_from_right, right_mesh
    )
    print(f"  canvas {out_w}x{out_h}", flush=True)
    if max(out_h, out_w) > max_canvas:
        raise RuntimeError(f"canvas {out_w}x{out_h} exceeds --max-canvas {max_canvas}")

    frames = []
    for index, (path_left, path_mid, path_right) in enumerate(frame_paths):
        images = [
            bgr_tensor(load_bgr(path_left, size_wh)),
            bgr_tensor(load_bgr(path_mid, size_wh)),
            bgr_tensor(load_bgr(path_right, size_wh)),
        ]
        meshes = [left_mesh[:, index], middle[:, index], right_mesh[:, index]]
        colors, valids = [], []
        for image, mesh in zip(images, meshes):
            color, valid = warp_frame(image, mesh, width_min, height_min, out_h, out_w, warp_mode)
            colors.append(color)
            valids.append(valid)
        fusion = linear_blender(colors[0], colors[1], valids[0], valids[1])
        cover12 = valids[0] + valids[1] - valids[0] * valids[1]
        fusion = linear_blender(fusion, colors[2], cover12, valids[2])
        valid_f = ((valids[0] > 0.5) | (valids[1] > 0.5) | (valids[2] > 0.5)).float()
        cam5_f = (valids[1] > 0.5).float()
        fusion, valid_f, cam5_f = inverse_to_cam5(
            fusion, valid_f, cam5_f, middle[:, index], width_min, height_min,
            img_h, img_w, warp_mode, max_canvas,
        )
        valid = (valid_f[0, 0] > 0.5).detach().cpu().numpy()
        cam5 = (cam5_f[0, 0] > 0.5).detach().cpu().numpy()
        image = fusion[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 255).astype(np.uint8)
        image[~valid] = 255
        frames.append((image, valid, cam5))
        del colors, valids, fusion, images
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(frame_paths):
            print(f"  warped {index + 1}/{len(frame_paths)}", flush=True)
    return frames


def is_oom(exc):
    return "out of memory" in str(exc).lower()


def other_process_holds_gpu(exc=None, minimum_mib=1024):
    me = os.getpid()
    if exc is not None:
        for pid, amount, unit in re.findall(r"Process (\d+) has ([0-9.]+) ([MG]iB)", str(exc)):
            if int(pid) == me:
                continue
            mib = float(amount) * (1024 if unit.startswith("G") else 1)
            if mib >= minimum_mib:
                return True
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        text = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, used = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid != me and used >= minimum_mib:
            return True
    return False


def oom_kind(exc):
    if not is_oom(exc):
        return None
    return "contention" if other_process_holds_gpu(exc) else "method"


def camera_on_white(image):
    height, width = image.shape[:2]
    canvas = np.full((height, width * 3, 3), 255, np.uint8)
    canvas[:, width:width * 2] = image
    valid = np.zeros((height, width * 3), dtype=bool)
    valid[:, width:width * 2] = True
    return canvas, valid, valid.copy()


def crop_cam5_triple(image, cam5):
    """Camera 5 in the center, with one camera-5 width on each side.

    Height is camera 5's bounding box. Width is three times that box, so the
    metric view is the restored wide field centered on the original camera.
    Pixels outside the panorama stay white.
    """
    ys, xs = np.where(cam5)
    if len(xs) == 0:
        raise RuntimeError("camera 5 mask is empty")
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    ref_h = y1 - y0
    ref_w = x1 - x0
    if ref_h < 1 or ref_w < 1:
        raise RuntimeError("camera 5 rectangle is empty")
    output = np.full((ref_h, ref_w * 3, image.shape[2]), 255, dtype=image.dtype)
    src_x0 = x0 - ref_w
    src_y0 = y0
    x_from = max(0, src_x0)
    y_from = max(0, src_y0)
    x_to = min(image.shape[1], src_x0 + ref_w * 3)
    y_to = min(image.shape[0], src_y0 + ref_h)
    if x_to > x_from and y_to > y_from:
        output[y_from - src_y0:y_to - src_y0, x_from - src_x0:x_to - src_x0] = image[y_from:y_to, x_from:x_to]
    return output


def resize_to(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    shrinking = image.shape[1] > width or image.shape[0] > height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def save_metric(image, valid, cam5, gt_w, gt_h, preview_dir, metric_dir, frame):
    if max(image.shape[:2]) > 65500:
        raise RuntimeError(f"canvas {image.shape[1]}x{image.shape[0]} exceeds the image limit")
    stitch = image.copy()
    stitch[~valid] = 255
    cropped = crop_cam5_triple(stitch, cam5)
    resized = resize_to(cropped, gt_w, gt_h)
    stitch_dir = os.path.join(preview_dir, "stitch")
    os.makedirs(stitch_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    cv2.imwrite(os.path.join(stitch_dir, f"{frame}.jpg"), stitch)
    cv2.imwrite(os.path.join(metric_dir, f"{frame}.png"), resized)
    return stitch.shape, resized.shape


def outputs_ready(preview_dir, metric_dir, frame):
    return os.path.isfile(os.path.join(preview_dir, "stitch", f"{frame}.jpg")) and os.path.isfile(
        os.path.join(metric_dir, f"{frame}.png")
    )


def target_size(sample_path, max_height):
    image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {sample_path}")
    height, width = image.shape[:2]
    if height > max_height:
        scale = float(max_height) / float(height)
        return (max(1, int(round(width * scale))), max_height)
    return (width, height)


def main():
    args = parse_args()
    spec = GT_SPECS[args.dataset]
    gt_root = resolve_data(spec["root"])
    gt_w, gt_h = spec["size"]
    if args.gt_width is not None:
        gt_w = args.gt_width
    if args.gt_height is not None:
        gt_h = args.gt_height
    if args.all:
        scenes = read_list(resolve_data(spec["val_list"]))
    elif args.scene:
        scenes = [args.scene]
    else:
        raise SystemExit("pass --scene or --all")

    spatial_net, temporal_net, smooth_net = load_nets()
    method = "threeview"
    print(f"{args.dataset}: {len(scenes)} scenes -> {gt_w}x{gt_h}, method {method}", flush=True)
    saved = skipped = failed = 0
    for scene_index, scene_name in enumerate(scenes):
        scene = scene_dir(args.dataset, scene_name)
        gt_dir = gt_root / scene.name
        frames = list_video_frames(scene)
        if args.frame:
            wanted = {frame_token(frame) for frame in args.frame}
            present = [frame for frame in frames if frame in wanted]
            missing = sorted(wanted - set(present))
            if missing:
                print(f"  scene {scene.name}: missing frames {missing}", flush=True)
            if not present:
                failed += 1
                continue
            # Frames before the smooth window are defined by the clip prefix, same as the official script.
            last = max(frames.index(frame) for frame in present)
            frames = frames[: last + 1]
            save_ids = set(present)
        else:
            if args.max_frames > 0:
                frames = frames[: args.max_frames]
            save_ids = set(frames)
        save_ids = {frame for frame in save_ids if has_gt(gt_dir, frame, spec["suffix"])}
        if len(frames) < SMOOTH_WINDOW:
            print(f"scene {scene.name}: only {len(frames)} frames, need {SMOOTH_WINDOW}", flush=True)
            failed += 1
            continue
        preview_dir = os.path.join(args.output_dir, args.dataset, scene.name, method)
        metric_dir = os.path.join(args.metric_dir, args.dataset, method, scene.name)
        pending = [frame for frame in frames if frame in save_ids and not outputs_ready(preview_dir, metric_dir, frame)]
        if not pending:
            skipped += len(save_ids)
            print(f"scene {scene_index + 1}/{len(scenes)} {scene.name}: already done", flush=True)
            continue
        print(f"scene {scene_index + 1}/{len(scenes)} {scene.name}: {len(frames)} video frames, save {len(pending)}", flush=True)
        image_dir = scene / "images"
        paths = [
            (image_dir / f"{frame}_{CAM_LEFT}.jpg", image_dir / f"{frame}_{CAM_MID}.jpg", image_dir / f"{frame}_{CAM_RIGHT}.jpg")
            for frame in frames
        ]
        try:
            size_wh = (518, 434) if "1224" in args.dataset else (518, 294)
            stitched = stitch_clip(
                spatial_net, temporal_net, smooth_net, paths, size_wh, args.warp_mode, args.max_canvas
            )
        except (RuntimeError, OSError, ValueError) as exc:
            torch.cuda.empty_cache()
            if oom_kind(exc) == "contention":
                print(f"  {scene.name}: another process holds the GPU, keeping camera 5 on white", flush=True)
                for frame in pending:
                    camera = load_bgr(image_dir / f"{frame}_{CAM_MID}.jpg", size_wh)
                    image, valid, cam5 = camera_on_white(camera)
                    save_metric(image, valid, cam5, gt_w, gt_h, preview_dir, metric_dir, frame)
                    saved += 1
                continue
            failed += len(pending)
            if oom_kind(exc) == "method":
                print(f"  fail {scene.name}: method exceeds GPU memory, nothing saved", flush=True)
            else:
                print(f"  skip {scene.name}: {exc}", flush=True)
            continue
        # One canvas covers the whole clip, so a partial rerun rewrites every saved frame.
        for frame, (image, valid, cam5) in zip(frames, stitched):
            if frame not in save_ids:
                continue
            try:
                stitch_shape, metric_shape = save_metric(
                    image, valid, cam5, gt_w, gt_h, preview_dir, metric_dir, frame
                )
            except (RuntimeError, OSError, ValueError) as exc:
                failed += 1
                print(f"  skip {scene.name} {frame}: {exc}", flush=True)
                continue
            saved += 1
            if saved == 1:
                print(
                    f"  {frame}: stitch {stitch_shape[1]}x{stitch_shape[0]} crop {metric_shape[1]}x{metric_shape[0]}",
                    flush=True,
                )
        torch.cuda.empty_cache()
    print(f"{method}: saved {saved}, skipped {skipped}, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
