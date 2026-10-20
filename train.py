#!/usr/bin/env python3
"""Run StabStitch++ driving finetuning as spatial, temporal, then smooth.

    python train.py --dataset widedrive3 --gpu 4 --epochs 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, args, extra):
    command = [
        sys.executable, "-u", str(ROOT / script),
        "--dataset", args.dataset,
        "--gpu", args.gpu,
        "--epochs", str(args.epochs),
        "--workers", str(args.workers),
        *extra,
    ]
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    print(f"\n########## {script} ##########", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="StabStitch spatial, temporal, then smooth")
    parser.add_argument("--dataset", required=True, choices=("nuscenes", "lyft", "widedrive3"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage", choices=("both", "spatial", "temporal", "smooth"), default="both")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smooth-batch", type=int, default=1, help="one 7-frame clip; 8 does not fit a 24GB GPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.stage in ("both", "spatial"):
        run("finetune_drive.py", args, ["--batch-size", str(args.batch_size)])
    if args.stage in ("both", "temporal"):
        run("finetune_temporal.py", args, ["--batch-size", str(args.batch_size)])
    if args.stage in ("both", "smooth"):
        run("finetune_smooth.py", args, ["--batch-size", str(args.smooth_batch)])


if __name__ == "__main__":
    main()
