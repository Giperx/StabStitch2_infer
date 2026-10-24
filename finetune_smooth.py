#!/usr/bin/env python3
"""Finetune StabStitch++ smooth warp from online spatial and temporal motions.

A sample is seven consecutive frames of camera 5 and one side camera. Frozen
spatial and temporal networks produce the motion tensors the original smooth
trainer reads from disk. One 7-frame window is trained with the original
data, smoothness, shape, trajectory, and alignment losses.
"""
import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--gpu", default="0")
_PRE_ARGS, _ = _PRE.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_PRE_ARGS.gpu)

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def stable_step(optimizer, parameters, max_growth=3.0):
    params = [p for p in parameters if p.requires_grad]
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            optimizer.zero_grad(set_to_none=True)
            return "nonfinite grad"
    with torch.no_grad():
        before = torch.zeros((), device=params[0].device)
        saved = [p.detach().clone() for p in params]
        for p in params:
            before = before + p.detach().float().pow(2).sum()
        before = torch.sqrt(before)
    optimizer.step()
    with torch.no_grad():
        bad = any(not torch.isfinite(p).all() for p in params)
        after = torch.zeros((), device=params[0].device)
        if not bad:
            for p in params:
                after = after + p.detach().float().pow(2).sum()
            after = torch.sqrt(after)
            bad = float(before) > 0 and float(after) > max_growth * float(before)
        if bad:
            for p, old in zip(params, saved):
                p.copy_(old)
            optimizer.zero_grad(set_to_none=True)
            return "weight x{:.1f}".format(float(after) / max(float(before), 1e-12))
    return ""

ROOT = Path(__file__).resolve().parent
_LEGACY = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub/datasets")
WORK = _LEGACY if _LEGACY.exists() else ROOT.parent / "datasets"
WINDOW = 7


def camera_size(dataset):
    return (518, 434) if "1224" in dataset else (518, 294)


EPOCHS = {"nuscenes": 1, "lyft": 3, "lyft1920": 3, "lyft1224": 3, "widedrive3": 3}
PARTS = {
    "nuscenes": [(WORK / "nuscenes/processed_10Hz_v2", "nuScenes_Train.txt")],
    "lyft1920": [(WORK / "lyft/lyft_train1920_3cams", "train1920.txt")],
    "lyft1224": [(WORK / "lyft/lyft_train1224_3cams", "train1224.txt")],
    "lyft": [
        (WORK / "lyft/lyft_train1920_3cams", "train1920.txt"),
        (WORK / "lyft/lyft_train1224_3cams", "train1224.txt"),
    ],
    "widedrive3": [(WORK / "WideDrive3_processed/WideDriveTrain", "train.txt")],
}


def unload_shared():
    for key in list(sys.modules):
        if key in {"utils", "grid_res", "loss"} or key.startswith(("utils.", "grid_res.")):
            del sys.modules[key]


def load_module(name, path):
    unload_shared()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent = str(path.parent)
    sys.path.insert(0, parent)
    spec.loader.exec_module(module)
    return module


def state_of(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def read_list(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_image(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
    array = image.astype(np.float32)
    array = (array / 127.5) - 1.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


class Clips(Dataset):
    def __init__(self, parts, size):
        self.size = size
        self.clips = []
        for root, name in parts:
            for scene in read_list(root / name):
                image_dir = root / scene / "images"
                if not image_dir.is_dir():
                    continue
                frames = {}
                for path in image_dir.glob("*.jpg"):
                    frame, cam = path.stem.rsplit("_", 1)
                    if cam in {"3", "4", "5"}:
                        frames.setdefault(int(frame), set()).add(cam)
                ordered = sorted(frame for frame, cams in frames.items() if {"3", "4", "5"} <= cams)
                size = camera_size(str(root))
                for start in range(0, len(ordered) - WINDOW + 1):
                    chunk = ordered[start:start + WINDOW]
                    if chunk[-1] - chunk[0] == WINDOW - 1:
                        self.clips.append((image_dir, chunk, size))
        if not self.clips:
            raise SystemExit("no 7-frame clips")
        print(f"smooth clips {len(self.clips)}", flush=True)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        image_dir, frames, size = self.clips[index]
        self.size = size
        side = 4 if random.randint(0, 1) else 3
        left = torch.stack([load_image(image_dir / f"{frame:03d}_5.jpg", self.size) for frame in frames])
        right = torch.stack([load_image(image_dir / f"{frame:03d}_{side}.jpg", self.size) for frame in frames])
        return left, right


def motions(spatial, temporal, left, right):
    spatial_motions = []
    temporal_motions = []
    with torch.no_grad():
        for index in range(left.shape[0]):
            out = spatial.build_model(spatial.net, left[index:index + 1], right[index:index + 1], is_training=False)
            spatial_motions.append((out["mesh_ref"] - out["mesh_rigid"], out["mesh_tgt"] - out["mesh_rigid"]))
            if index == 0:
                zeros = torch.zeros_like(spatial_motions[-1][0])
                temporal_motions.append((zeros, zeros.clone()))
            else:
                ref = temporal.build_model(temporal.net, left[index - 1:index], left[index:index + 1], is_training=False)
                tgt = temporal.build_model(temporal.net, right[index - 1:index], right[index:index + 1], is_training=False)
                temporal_motions.append((ref["motion"], tgt["motion"]))
    return temporal_motions, spatial_motions


def window_loss(loss_fn, out):
    data = loss_fn.l_num_loss(out["ori_path1"], out["smooth_path1"], 2)
    data = data + loss_fn.l_num_loss(out["ori_path2"], out["smooth_path2"], 2)
    shape = loss_fn.inter_grid_loss(out["target_mesh1"]) + loss_fn.intra_grid_loss(out["target_mesh1"])
    shape = shape + loss_fn.inter_grid_loss(out["target_mesh2"]) + loss_fn.intra_grid_loss(out["target_mesh2"])
    trajectory = loss_fn.l_num_loss(out["dense_spath1"] * out["ovmask_spath"], out["dense_spath2"] * out["ovmask_spath"], 1)
    align = loss_fn.cal_lp_loss(out["wimg1"], out["wimg2"], out["ovmask_img"])
    path = out["smooth_path1"]
    smooth = loss_fn.l_num_loss(path[:, :-6], path[:, 3:-3], 2) + loss_fn.l_num_loss(path[:, 6:], path[:, 3:-3], 2)
    return data + 50 * smooth + 10 * shape + trajectory + 1000 * align


def main():
    parser = argparse.ArgumentParser(description="Finetune smooth warp")
    parser.add_argument("--dataset", required=True, choices=sorted(EPOCHS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--spatial", default="")
    parser.add_argument("--temporal", default="")
    parser.add_argument("--weight", default=str(ROOT / "pretrained" / "smooth_warp.pth"))
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    epochs = args.epochs or EPOCHS[args.dataset]
    spatial_path = args.spatial or str(ROOT / "SpatialWarp" / "model_drive" / args.dataset / "spatial_warp.pth")
    if not Path(spatial_path).is_file():
        spatial_path = str(ROOT / "pretrained" / "spatial_warp.pth")
    temporal_path = args.temporal or str(ROOT / "TemporalWarp" / "model_drive" / args.dataset / "temporal_warp.pth")
    if not Path(temporal_path).is_file():
        temporal_path = str(ROOT / "pretrained" / "temporal_warp.pth")

    spatial = load_module("stab_spatial", ROOT / "SpatialWarp" / "Codes" / "network.py")
    temporal = load_module("stab_temporal", ROOT / "TemporalWarp" / "Codes" / "network.py")
    smooth = load_module("stab_smooth", ROOT / "SmoothWarp" / "Codes" / "network.py")
    loss_fn = load_module("stab_smooth_loss", ROOT / "SmoothWarp" / "Codes" / "loss.py")
    spatial.net = spatial.Network().cuda().eval()
    temporal.net = temporal.Network().cuda().eval()
    spatial.net.load_state_dict(state_of(spatial_path))
    temporal.net.load_state_dict(state_of(temporal_path))
    for parameter in list(spatial.net.parameters()) + list(temporal.net.parameters()):
        parameter.requires_grad_(False)
    net = smooth.Network().cuda().train()
    net.load_state_dict(state_of(args.weight))
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    loader = DataLoader(Clips(PARTS[args.dataset], camera_size(args.dataset)), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    save_dir = ROOT / "SmoothWarp" / "model_drive" / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"smooth from {args.weight}; spatial {spatial_path}; temporal {temporal_path}", flush=True)
    steps = 0
    for epoch in range(epochs):
        for index, (left, right) in enumerate(loader):
            # One clip per step. A batch stacks clips as [batch, time, c, h, w].
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for item in range(left.shape[0]):
                view1 = left[item].cuda(non_blocking=True)
                view2 = right[item].cuda(non_blocking=True)
                temporal_motions, spatial_motions = motions(spatial, temporal, view1, view2)
                out = smooth.build_model(
                    net,
                    [item[0] for item in temporal_motions],
                    [item[1] for item in temporal_motions],
                    [item[0] for item in spatial_motions],
                    [item[1] for item in spatial_motions],
                    [frame.unsqueeze(0) for frame in view1],
                    [frame.unsqueeze(0) for frame in view2],
                )
                losses.append(window_loss(loss_fn, out))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                print(f"skip smooth step {index + 1}: nonfinite loss", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            reason = stable_step(optimizer, net.parameters())
            if reason:
                print(f"skip smooth step {index + 1}: {reason}", flush=True)
                continue
            steps += 1
            print(f"smooth epoch {epoch + 1}/{epochs} step {index + 1} loss {float(loss):.4f}", flush=True)
            if args.max_steps and steps >= args.max_steps:
                path = save_dir / "smooth_warp.pth"
                torch.save({"model": net.state_dict(), "epoch": epoch + 1}, path)
                print(f"stopped after {steps} steps, saved {path}", flush=True)
                return
        path = save_dir / f"epoch_{epoch + 1}.pth"
        torch.save({"model": net.state_dict(), "epoch": epoch + 1}, path)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
