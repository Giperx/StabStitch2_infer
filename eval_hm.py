#!/usr/bin/env python3
"""Dataset-level histogram-matched sparse-GT metrics for StabStitch++ threeview stitches.

This is a local reimplementation of the driving-baseline metric contract
(Residual-Elastic-Warp ``rewarp_drive/metrics.py`` and RopStitch
``eval_hm_dataset.py``). Nothing is imported from those repositories.

It scores the predictions written by ``stitch_drive.py``:

  {ROOT}/outputs/metric_eval/{dataset}/threeview/{scene}/{frame}.png

and writes:

  {ROOT}/outputs/metric_eval/{dataset}/threeview_hm.txt
  {ROOT}/outputs/metric_eval/lyft_threeview_hm.txt   (pooled lyft)

Contract:
  * Images are RGB float64 in ``[0, 1]``. The mask is the sparse GT mask
    (``L > 127``). The prediction is histogram-matched to the GT on the masked
    pixels only via ``skimage.exposure.match_histograms``.
  * ``mae = mean(|matched - target|)``, ``mse = mean(diff**2)``,
    ``rmse = sqrt(mse)``.
  * ``ssim`` is the per-pixel luminance term
    ``mean((2*m*t + c1) / (m**2 + t**2 + c1))`` with ``c1 = 0.01**2``.
  * ``psnr = 10*log10(1/mse)`` (``inf`` when ``mse == 0``).
  * MAE/RMSE are reported on the 0-255 scale, PSNR on ``[0, 1]``. No LPIPS.
  * Regions are the three equal widths of the canvas; each third is
    histogram-matched independently. L+R and L+R+C are unweighted means of the
    region means, not a second pixel pool.
  * Dataset aggregation pools every frame; the PSNR mean skips non-finite
    values. Full-image and each region are separate pools.

Examples:
  python eval_hm.py --dataset nuscenes
  python eval_hm.py --dataset lyft
"""
import argparse
import inspect
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.exposure import match_histograms

ROOT = Path(__file__).resolve().parent
WORK = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub")

REGION_ORDER = ("Left", "Center", "Right")

# scikit-image renamed the multichannel flag to ``channel_axis`` in 0.19; the
# repository environment may pin an older release. The result is identical, so
# use the modern call when available and fall back otherwise.
_HAS_CHANNEL_AXIS = "channel_axis" in inspect.signature(match_histograms).parameters


# ---------------------------------------------------------------------------
# Dataset registry (paths are relative to the data root)
# ---------------------------------------------------------------------------
DATASETS = {
    "nuscenes": {
        "gt": "nuscenes/sparseWideFOVImages3_1554x294",
        "cam": 5,
    },
    "lyft1920": {
        "gt": "lyft/1920_sparseWideFOVImages3_1554x294",
        "cam": 5,
    },
    "lyft1224": {
        "gt": "lyft/1224_sparseWideFOVImages3_1554x434",
        "cam": 5,
    },
    "widedrive3": {
        "gt": "WideDrive_processed/sparseWideFOVImages3_1554x294",
        "cam": 2,  # camera 2 is only the GT filename; stitching still uses 5/4/3
        "direct": True,
    },
}

LYFT_PARTS = ("lyft1920", "lyft1224")


def resolve_data(relative, data_root=None):
    """Local ``{ROOT}/datasets/<relative>`` first, else the shared dataset root."""
    relative = Path(relative)
    if data_root is not None:
        root = Path(data_root) / relative
        if not root.exists():
            raise SystemExit(f"dataset path not found: {root}")
        return root
    local = ROOT / "datasets" / relative
    if local.exists():
        return local
    shared = WORK / "datasets" / relative
    if shared.exists():
        return shared
    raise SystemExit(f"dataset path not found: {local} or {shared}")


# ---------------------------------------------------------------------------
# Core metric math
# ---------------------------------------------------------------------------
def frame_hm(render, gt, mask, match=True):
    """Histogram-matched scores, or direct scores when match is false."""
    if mask.sum() == 0:
        return None
    pred = render[mask]
    target = gt[mask]
    if not match:
        matched = pred
        diff = matched - target
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        c1 = 0.01 ** 2
        ssim = float(np.mean((2 * matched * target + c1) / (matched ** 2 + target ** 2 + c1)))
        return {
            "hm_mae": mae,
            "hm_rmse": float(np.sqrt(mse)),
            "hm_psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
            "hm_ssim": ssim,
        }
    if _HAS_CHANNEL_AXIS:
        matched = match_histograms(pred[np.newaxis, ...], target[np.newaxis, ...], channel_axis=-1)[0]
    else:
        matched = match_histograms(pred[np.newaxis, ...], target[np.newaxis, ...], multichannel=True)[0]
    diff = matched - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    c1 = 0.01 ** 2
    ssim = float(np.mean((2 * matched * target + c1) / (matched ** 2 + target ** 2 + c1)))
    return {
        "hm_mae": mae,
        "hm_rmse": float(np.sqrt(mse)),
        "hm_psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "hm_ssim": ssim,
    }


def find_gt(gt_dir, frame, cam):
    """Locate ``{gt_dir}/rgb/{frame}_{cam}_sparse_wide.png`` and its mask."""
    suffix = f"_{cam}_sparse_wide.png"
    rgb = os.path.join(str(gt_dir), "rgb", f"{frame}{suffix}")
    mask = os.path.join(str(gt_dir), "mask", f"{frame}{suffix}")
    if os.path.exists(rgb) and os.path.exists(mask):
        return rgb, mask
    return None, None


def match_gt(render, gt, mask):
    """Resize GT (bilinear) and mask (nearest) to the render size when needed."""
    if gt.shape[:2] == render.shape[:2] and mask.shape == render.shape[:2]:
        return gt, mask
    size = (render.shape[1], render.shape[0])
    gt_img = Image.fromarray(np.clip(gt * 255.0, 0, 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(size, Image.NEAREST)
    return np.array(gt_img).astype(np.float64) / 255.0, np.array(mask_img) > 127


def region_slices(width):
    """Three equal-width column slices. Width not divisible by 3 is an error."""
    if width % 3 != 0:
        raise SystemExit(f"width {width} is not divisible by 3")
    orig_w = width // 3
    return {
        "Left": (0, orig_w),
        "Center": (orig_w, orig_w * 2),
        "Right": (orig_w * 2, orig_w * 3),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean_rows(rows):
    """Mean over frames. The PSNR mean skips non-finite values."""
    if not rows:
        return None
    return {
        "hm_psnr": float(np.mean([row["hm_psnr"] for row in rows if np.isfinite(row["hm_psnr"])])),
        "hm_ssim": float(np.mean([row["hm_ssim"] for row in rows])),
        "hm_mae": float(np.mean([row["hm_mae"] for row in rows])),
        "hm_rmse": float(np.mean([row["hm_rmse"] for row in rows])),
        "n": len(rows),
    }


def mean_of(stats):
    """Unweighted mean of per-region means."""
    keys = ("hm_psnr", "hm_ssim", "hm_mae", "hm_rmse")
    return {key: float(np.mean([item[key] for item in stats])) for key in keys}


def format_stats(stats, frames=True, lpips=False):
    text = (
        f"HM-PSNR: {stats['hm_psnr']:.2f}, HM-SSIM: {stats['hm_ssim']:.4f}, "
        f"HM-MAE: {stats['hm_mae'] * 255:.2f}, HM-RMSE: {stats['hm_rmse'] * 255:.2f}"
    )
    if lpips and "lpips" in stats:
        text += f", LPIPS: {stats['lpips']:.4f}"
    if frames and "n" in stats:
        text += f", Frames: {stats['n']}"
    return text


def load_lpips(device):
    """AlexNet spatial LPIPS, the same network as the WideDrive dense metrics."""
    import lpips

    model = lpips.LPIPS(net="alex", spatial=True).to(device)
    model.eval()
    return model


def full_lpips(lpips_fn, render, gt, device):
    """Mean spatial LPIPS over every pixel. WideDrive GT is dense, so no mask."""
    pred = torch.from_numpy(np.ascontiguousarray(render)).float().permute(2, 0, 1).unsqueeze(0)
    ref = torch.from_numpy(np.ascontiguousarray(gt)).float().permute(2, 0, 1).unsqueeze(0)
    pred = pred.to(device) * 2.0 - 1.0
    ref = ref.to(device) * 2.0 - 1.0
    with torch.no_grad():
        lpips_map = lpips_fn(pred, ref)
    return float(lpips_map.detach().float().mean().cpu())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def pred_root_for(output_root, dataset, method):
    return Path(output_root) / "metric_eval" / dataset / method


def default_report_path(output_root, dataset, method):
    return Path(output_root) / "metric_eval" / dataset / f"{method}_hm.txt"


def blank_side(render, fraction=0.90, level=0.98):
    image = render.astype(np.float64)
    if image.max(initial=0) > 1.5:
        image = image / 255.0
    white = image.min(axis=-1) >= level
    third = white.shape[1] // 3
    if third < 1:
        return True
    for start in (0, white.shape[1] - third):
        if float(white[:, start:start + third].mean()) >= fraction:
            return True
    return False


def score_spec(spec, pred_root, gt_root, lpips_fn=None, device=None):
    """Score every ``{scene}/{frame}.png`` under ``pred_root`` for one dataset.

    ``gt_root`` is already the sparse-GT directory (the parent of the scene
    folders). ``resolve_data`` has applied ``spec['gt']``. WideDrive also gets a
    full-image LPIPS: its GT is dense, so that score uses every pixel.
    """
    pred_root = Path(pred_root)
    if not pred_root.is_dir():
        raise SystemExit(f"no predictions under {pred_root}")
    scenes = sorted(name for name in os.listdir(pred_root) if os.path.isdir(pred_root / name))
    full_rows = []
    region_rows = {name: [] for name in REGION_ORDER}
    per_scene = []
    skipped = 0
    blank = 0
    for scene in scenes:
        pred_dir = pred_root / scene
        gt_dir = Path(gt_root) / scene
        names = sorted(name for name in os.listdir(pred_dir) if name.lower().endswith(".png"))
        scene_full = []
        scene_regions = {name: [] for name in REGION_ORDER}
        for name in names:
            frame = os.path.splitext(name)[0]
            gt_path, mask_path = find_gt(gt_dir, frame, spec["cam"])
            if gt_path is None:
                skipped += 1
                continue
            render = np.array(Image.open(pred_dir / name).convert("RGB")).astype(np.float64) / 255.0
            if blank_side(render):
                blank += 1
                print(f"  reject {scene} {frame}: blank side", flush=True)
                continue
            gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float64) / 255.0
            mask = np.array(Image.open(mask_path).convert("L")) > 127
            gt, mask = match_gt(render, gt, mask)
            direct = spec.get("direct")
            if direct:
                mask = np.ones(render.shape[:2], dtype=bool)
            row = frame_hm(render, gt, mask, match=not direct)
            if row is None:
                skipped += 1
                continue
            if lpips_fn is not None:
                row["lpips"] = full_lpips(lpips_fn, render, gt, device)
            row["frame"] = frame
            scene_full.append(row)
            slices = region_slices(render.shape[1])
            for region, (x0, x1) in slices.items():
                region_row = frame_hm(render[:, x0:x1], gt[:, x0:x1], mask[:, x0:x1], match=not spec.get("direct"))
                if region_row is not None:
                    region_row["frame"] = frame
                    scene_regions[region].append(region_row)
        if scene_full:
            full_rows.extend(scene_full)
            for region in REGION_ORDER:
                region_rows[region].extend(scene_regions[region])
            scene_stats = mean_rows(scene_full)
            if lpips_fn is not None and "lpips" in scene_full[0]:
                scene_stats["lpips"] = float(np.mean([row["lpips"] for row in scene_full if np.isfinite(row["lpips"])]))
            per_scene.append(
                (scene, scene_stats, {name: mean_rows(rows) for name, rows in scene_regions.items()})
            )
            print(f"  {scene}: {len(scene_full)} frames", flush=True)
    judged = len(full_rows) + blank
    print(
        f"Success: {len(full_rows)}/{judged} ({(len(full_rows) / judged if judged else 0):.1%}); "
        f"blank-side rejected: {blank}",
        flush=True,
    )
    return full_rows, region_rows, per_scene, skipped


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report_lines(title, full_rows, region_rows, per_scene, skipped, extra, lpips=False):
    full_stats = mean_rows(full_rows)
    if lpips and full_stats is not None and full_rows and "lpips" in full_rows[0]:
        full_stats["lpips"] = float(np.mean([row["lpips"] for row in full_rows if np.isfinite(row["lpips"])]))
    region_stats = {name: mean_rows(region_rows[name]) for name in REGION_ORDER}
    lines = [
        f"=== {title} HM ===",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        *extra,
        "Mask: sparse GT mask only. HM scores ignore pixels outside that mask.",
        "HM-SSIM is the per-pixel luminance term.",
        "L+R and L+R+C are unweighted means of the region means.",
        "MAE and RMSE are on the 0-255 scale. PSNR uses [0, 1] images.",
        f"Skipped frames: {skipped}.",
        "",
        "Full",
        format_stats(full_stats, lpips=lpips) if full_stats else "no frames",
        "",
        "Regions",
    ]
    for name in REGION_ORDER:
        stats = region_stats[name]
        lines.append(f"{name:8s}: {format_stats(stats) if stats else 'no frames'}")
    if all(region_stats[name] for name in ("Left", "Right")):
        lines.append("L+R    : " + format_stats(mean_of([region_stats["Left"], region_stats["Right"]]), frames=False))
    if all(region_stats.values()):
        lines.append("L+R+C  : " + format_stats(mean_of(list(region_stats.values())), frames=False))
    lines.append("")
    if lpips:
        lines.append("LPIPS is AlexNet, mean over every pixel of the full image. WideDrive GT is dense.")
    lines.append("Per-scene Full")
    for scene, scene_stats, _regions in per_scene:
        lines.append(f"{scene}: {format_stats(scene_stats, lpips=lpips)}")
    return lines


def write_report(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(path, "w") as handle:
        handle.write(text)
    print(text, end="")
    print(f"Wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(dataset, output_root=None, data_root=None, report_path=None, method="threeview"):
    """Score one dataset (or pooled ``lyft``) and write the HM report."""
    output_root = Path(output_root) if output_root is not None else ROOT / "outputs"
    want_lpips = dataset == "widedrive3"
    lpips_fn = device = None
    if want_lpips:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = load_lpips(device)

    if dataset == "lyft":
        parts = []
        for name in LYFT_PARTS:
            print(f"scoring {name}", flush=True)
            pred_root = pred_root_for(output_root, name, method)
            spec = DATASETS[name]
            gt_root = resolve_data(spec["gt"], data_root)
            parts.append((name, score_spec(spec, pred_root, gt_root, lpips_fn, device)))
        full_rows = [row for _name, (rows, _regions, _scenes, _skip) in parts for row in rows]
        if not full_rows:
            raise SystemExit(f"no scored frames for lyft under {output_root}/metric_eval")
        region_rows = {name: [] for name in REGION_ORDER}
        per_scene = []
        skipped = 0
        extra = ["Pool: every lyft1920 and lyft1224 frame. Subsets are not averaged."]
        for name, (rows, regions, scenes, skip) in parts:
            skipped += skip
            for region in REGION_ORDER:
                region_rows[region].extend(regions[region])
            stats = mean_rows(rows)
            extra.append(f"{name}: {format_stats(stats) if stats else 'no frames'}")
            per_scene.extend(
                (f"{name}/{scene}", full_stats, region_stats) for scene, full_stats, region_stats in scenes
            )
        path = (
            Path(report_path)
            if report_path is not None
            else Path(output_root) / "metric_eval" / f"lyft_{method}_hm.txt"
        )
        write_report(path, report_lines(f"lyft {method}", full_rows, region_rows, per_scene, skipped, extra, want_lpips))
        return path

    spec = DATASETS[dataset]
    print(f"scoring {dataset} {method}", flush=True)
    pred_root = pred_root_for(output_root, dataset, method)
    gt_root = resolve_data(spec["gt"], data_root)
    full_rows, region_rows, per_scene, skipped = score_spec(spec, pred_root, gt_root, lpips_fn, device)
    if not full_rows:
        raise SystemExit(f"no scored frames under {pred_root}")
    extra = [f"Pred: {pred_root}", f"GT: {gt_root}", f"GT camera: {spec['cam']}"]
    path = Path(report_path) if report_path is not None else default_report_path(output_root, dataset, method)
    write_report(path, report_lines(f"{dataset} {method}", full_rows, region_rows, per_scene, skipped, extra, want_lpips))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dataset-level sparse-GT HM metrics for StabStitch++ threeview stitches")
    parser.add_argument("--dataset", required=True, choices=[*DATASETS, "lyft"])
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs"),
        help="root holding metric_eval/ (default: {ROOT}/outputs)",
    )
    parser.add_argument("--data-root", default=None, help="dataset root (defaults to local then shared)")
    parser.add_argument("--report", default=None, help="explicit report path")
    parser.add_argument("--method", default="threeview", help="prediction folder under metric_eval/{dataset}/")
    args = parser.parse_args(argv)
    run(args.dataset, args.output, args.data_root, args.report, args.method)


if __name__ == "__main__":
    main()
