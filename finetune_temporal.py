#!/usr/bin/env python3
"""Finetune StabStitch++ temporal warp on consecutive driving frames.

The original temporal trainer takes two frames from one camera. Here those
frames are adjacent frames of camera 5, 4, or 3. Initialization is
pretrained/temporal_warp.pth.
"""
import argparse
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "TemporalWarp" / "Codes"))
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss  # noqa: E402
from network import Network, build_model  # noqa: E402

_LEGACY = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub/datasets")
WORK = _LEGACY if _LEGACY.exists() else ROOT.parent / "datasets"
PRETRAINED = ROOT / "pretrained" / "temporal_warp.pth"


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


def read_list(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class ConsecutiveFrames(Dataset):
    def __init__(self, parts, size):
        self.size = size
        self.pairs = []
        for root, name in parts:
            for scene in read_list(root / name):
                image_dir = root / scene / "images"
                if not image_dir.is_dir():
                    continue
                frames = {}
                for path in image_dir.glob("*.jpg"):
                    frame, cam = path.stem.rsplit("_", 1)
                    if cam in {"3", "4", "5"}:
                        frames.setdefault(cam, {})[int(frame)] = path
                size = camera_size(str(root))
                for cam, items in frames.items():
                    ordered = sorted(items)
                    for left, right in zip(ordered, ordered[1:]):
                        self.pairs.append((str(items[left]), str(items[right]), size))
        if not self.pairs:
            raise SystemExit("no consecutive frames")
        print(f"temporal pairs {len(self.pairs)}", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first, second, size = self.pairs[index]
        if random.randint(0, 1):
            first, second = second, first
        return load(first, size), load(second, size)


def load(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
    array = image.astype(np.float32)
    array = (array / 127.5) - 1.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def main():
    parser = argparse.ArgumentParser(description="Finetune temporal warp")
    parser.add_argument("--dataset", required=True, choices=sorted(EPOCHS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--weight", default=str(PRETRAINED))
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    epochs = args.epochs or EPOCHS[args.dataset]
    loader = DataLoader(
        ConsecutiveFrames(PARTS[args.dataset], camera_size(args.dataset)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    net = Network().cuda()
    checkpoint = torch.load(args.weight, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    net.load_state_dict(state)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    save_dir = ROOT / "TemporalWarp" / "model_drive" / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    steps = 0
    for epoch in range(epochs):
        for index, (left, right) in enumerate(loader):
            left = left.float().cuda(non_blocking=True)
            right = right.float().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = build_model(net, left, right)
            overlap = cal_lp_loss(left, out["output_mesh"])
            regular = 5 * inter_grid_loss(out["mesh"]) + 5 * intra_grid_loss(out["mesh"])
            loss = overlap + regular
            if not torch.isfinite(loss):
                print(f"skip temporal step {index + 1}: nonfinite loss", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            reason = stable_step(optimizer, net.parameters())
            if reason:
                print(f"skip temporal step {index + 1}: {reason}", flush=True)
                continue
            steps += 1
            if index % 20 == 0:
                print(
                    f"temporal epoch {epoch + 1}/{epochs} step {index + 1} "
                    f"loss {float(loss):.4f} overlap {float(overlap):.4f}",
                    flush=True,
                )
            if args.max_steps and steps >= args.max_steps:
                path = save_dir / "temporal_warp.pth"
                torch.save({"model": net.state_dict(), "epoch": epoch + 1}, path)
                print(f"stopped after {steps} steps, saved {path}", flush=True)
                return
        path = save_dir / f"epoch_{epoch + 1}.pth"
        torch.save({"model": net.state_dict(), "epoch": epoch + 1}, path)
        print(f"saved {path}", flush=True)
    final = save_dir / "temporal_warp.pth"
    torch.save({"model": net.state_dict(), "epoch": epochs}, final)
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
