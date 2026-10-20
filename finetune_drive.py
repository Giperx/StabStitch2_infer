#!/usr/bin/env python3
"""Finetune the StabStitch++ spatial warp on driving camera pairs.

The original spatial trainer (SpatialWarp/Codes/train_tra.py) takes two images.
Here those two images are camera 5 with camera 4, or camera 5 with camera 3.
Every such pair in the train scenes is one sample. DataLoader shuffle mixes
scenes and frames. The two views are also swapped at random, as in the
original dataset.

Initialization is the released spatial_warp.pth. The optimizer starts fresh.
Defaults: nuScenes 1 epoch, Lyft 1920 and 1224 mixed for 3 epochs, WideDrive3
3 epochs. WideDriveTrain is the sibling of WideDriveVal and uses train.txt.

    python finetune_drive.py --dataset nuscenes --gpu 0
    python finetune_drive.py --dataset lyft --gpu 0
    python finetune_drive.py --dataset widedrive3 --gpu 0
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--gpu", default="0")
_PRE_ARGS, _ = _PRE.parse_known_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(_PRE_ARGS.gpu)

import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "SpatialWarp" / "Codes"))

from loss import cal_lp_loss, cal_perception_loss, inter_grid_loss, intra_grid_loss  # noqa: E402
from network import Network, build_model  # noqa: E402

_LEGACY = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub/datasets")
WORK = _LEGACY if _LEGACY.exists() else ROOT.parent / "datasets"
PRETRAINED = ROOT / "pretrained" / "spatial_warp.pth"
WIDTH = 480
HEIGHT = 360

# Epochs are the requested finetune lengths. lr is lower than the original
# 1e-4 because these runs start from the released weights.
DATASETS = {
    "nuscenes": {
        "epochs": 1,
        "lr": 1e-5,
        "parts": [
            {
                "root": WORK / "nuscenes/processed_10Hz_v2",
                "list": "nuScenes_Train.txt",
            }
        ],
    },
    "lyft": {
        "epochs": 3,
        "lr": 1e-5,
        "parts": [
            {"root": WORK / "lyft/lyft_train1920_3cams", "list": "train1920.txt"},
            {"root": WORK / "lyft/lyft_train1224_3cams", "list": "train1224.txt"},
        ],
    },
    "widedrive3": {
        "epochs": 3,
        "lr": 1e-5,
        "parts": [
            {
                "root": WORK / "WideDrive3_processed/WideDriveTrain",
                "list": "train.txt",
            }
        ],
    },
}
MID = 5
SIDES = (4, 3)


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


class PairDataset(Dataset):
    """One sample is camera 5 and one overlapping side camera, same frame."""

    def __init__(self, parts):
        self.pairs = []
        for part in parts:
            root = Path(part["root"])
            list_path = root / part["list"]
            if not list_path.is_file():
                raise SystemExit(f"train list not found: {list_path}")
            for scene in read_list(list_path):
                image_dir = root / scene / "images"
                if not image_dir.is_dir():
                    continue
                frames = []
                for name in os.listdir(image_dir):
                    if name.endswith(f"_{MID}.jpg"):
                        frames.append(name[: -len(f"_{MID}.jpg")])
                for frame in frames:
                    mid = image_dir / f"{frame}_{MID}.jpg"
                    for side in SIDES:
                        other = image_dir / f"{frame}_{side}.jpg"
                        if mid.is_file() and other.is_file():
                            self.pairs.append((str(mid), str(other)))
        if not self.pairs:
            raise SystemExit("no cam5/cam4 or cam5/cam3 training pairs")
        print(f"pairs {len(self.pairs)}", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first, second = self.pairs[index]
        if random.randint(0, 1) == 1:
            first, second = second, first
        return load_tensor(first), load_tensor(second)


def load_tensor(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    array = image.astype(np.float32)
    array = (array / 127.5) - 1.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(array))


def load_net(path):
    net = Network().cuda()
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    net.load_state_dict(state)
    print(f"loaded {path}", flush=True)
    return net


def train(args):
    spec = DATASETS[args.dataset]
    epochs = args.epochs if args.epochs is not None else spec["epochs"]
    if not torch.cuda.is_available():
        raise SystemExit("finetuning needs CUDA")
    torch.cuda.set_device(0)

    data = PairDataset(spec["parts"])
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    net = load_net(args.weight)
    net.train()
    vgg = models.vgg19(pretrained=True).cuda().eval()
    for parameter in vgg.parameters():
        parameter.requires_grad = False
    optimizer = optim.Adam(net.parameters(), lr=spec["lr"], betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    save_dir = ROOT / "SpatialWarp" / "model_drive" / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.dataset}: {epochs} epochs, {len(loader)} steps/epoch, batch {args.batch_size}, lr {spec['lr']}",
        flush=True,
    )
    steps = 0
    started = time.time()
    for epoch in range(epochs):
        loss_sum = overlap_sum = nonoverlap_sum = 0.0
        seen = 0
        for index, batch in enumerate(loader):
            left = batch[0].float().cuda(non_blocking=True)
            right = batch[1].float().cuda(non_blocking=True)
            optimizer.zero_grad()
            out = build_model(net, left, right)
            overlap = cal_lp_loss(
                out["output_H_ref"], out["output_H_tgt"], out["output_tps_ref"], out["output_tps_tgt"]
            )
            overlap = overlap + 0.001 * cal_perception_loss(
                vgg, out["output_H_ref"], out["output_H_tgt"], out["output_tps_ref"], out["output_tps_tgt"]
            )
            nonoverlap = 5 * inter_grid_loss(out["mesh_ref"]) + 5 * intra_grid_loss(out["mesh_ref"])
            nonoverlap = nonoverlap + 5 * inter_grid_loss(out["mesh_tgt"]) + 5 * intra_grid_loss(out["mesh_tgt"])
            loss = overlap + nonoverlap
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            loss_sum += float(loss.item())
            overlap_sum += float(overlap.item())
            nonoverlap_sum += float(nonoverlap.item())
            seen += 1
            steps += 1
            if index % args.log_every == 0:
                elapsed = time.time() - started
                print(
                    f"epoch {epoch + 1}/{epochs} step {index + 1}/{len(loader)} "
                    f"loss {loss_sum / seen:.4f} overlap {overlap_sum / seen:.4f} "
                    f"nonoverlap {nonoverlap_sum / seen:.4f} {steps / elapsed:.2f} step/s",
                    flush=True,
                )
            if args.max_steps and steps >= args.max_steps:
                print(f"stopped after {steps} steps", flush=True)
                return
        scheduler.step()
        path = save_dir / f"epoch_{epoch + 1}.pth"
        torch.save(
            {"model": net.state_dict(), "epoch": epoch + 1, "dataset": args.dataset},
            path,
        )
        print(f"saved {path} loss {loss_sum / max(seen, 1):.4f}", flush=True)
    final = save_dir / "spatial_warp.pth"
    torch.save({"model": net.state_dict(), "epoch": epochs, "dataset": args.dataset}, final)
    print(f"saved {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Finetune StabStitch++ spatial warp on driving pairs")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=None, help="override the dataset epoch count")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight", default=str(PRETRAINED))
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early, for a timing check")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
