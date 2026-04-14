"""
train_yolo.py — fine-tune YOLOv8 on the billiards ball dataset.

Trains from a YOLOv8 nano or small checkpoint on the YOLO dataset generated
by prepare_yolo_data.py + autolabel.py.  Saves the best checkpoint to
models/best.pt so run_full_v2.py can load it with --model models/best.pt.

Usage (local, CPU/GPU):
  python scripts/train_yolo.py
  python scripts/train_yolo.py --model-size m --epochs 100 --batch 16

Usage on PSC (called from run_train.slurm):
  python scripts/train_yolo.py \
      --data yolo_dataset/data.yaml \
      --model-size s \
      --epochs 150 \
      --batch 32 \
      --device 0 \
      --project runs/train \
      --name billiards_v1

After training the best checkpoint is copied to models/best.pt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Make sure billiards_engine imports work when called as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on billiards dataset")
    parser.add_argument("--data",        default="yolo_dataset/data.yaml", type=Path,
                        help="Path to data.yaml (default: yolo_dataset/data.yaml)")
    parser.add_argument("--model-size",  default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 variant: n=nano s=small m=medium (default: n)")
    parser.add_argument("--pretrained",  default=None, type=Path,
                        help="Start from this checkpoint instead of the ImageNet pretrained weights")
    parser.add_argument("--epochs",      default=100, type=int)
    parser.add_argument("--batch",       default=16,  type=int)
    parser.add_argument("--imgsz",       default=640, type=int)
    parser.add_argument("--device",      default="",
                        help="Device: '' = auto, '0' = first GPU, 'cpu'")
    parser.add_argument("--project",     default="runs/train", type=Path)
    parser.add_argument("--name",        default="billiards_v1")
    parser.add_argument("--workers",     default=4,   type=int)
    parser.add_argument("--patience",    default=30,  type=int,
                        help="Early-stopping patience in epochs (0 = disabled)")
    parser.add_argument("--out",         default="models/best.pt", type=Path,
                        help="Where to copy best.pt after training (default: models/best.pt)")
    args = parser.parse_args()

    # Validate data.yaml
    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        sys.exit(
            f"ERROR: data.yaml not found at {data_yaml}\n"
            "Run scripts/prepare_yolo_data.py and scripts/autolabel.py first."
        )

    # Import Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ERROR: ultralytics not installed.\n"
            "  pip install ultralytics"
        )

    # Choose starting checkpoint
    if args.pretrained and args.pretrained.exists():
        ckpt = str(args.pretrained)
        print(f"  Starting from custom checkpoint: {ckpt}")
    else:
        ckpt = f"yolov8{args.model_size}.pt"   # downloads if not cached
        print(f"  Starting from pretrained: {ckpt}")

    model = YOLO(ckpt)

    print(f"\n  Dataset  : {data_yaml}")
    print(f"  Epochs   : {args.epochs}  (patience={args.patience})")
    print(f"  Batch    : {args.batch}   imgsz={args.imgsz}")
    print(f"  Device   : '{args.device}' (blank = auto)")
    print(f"  Output   : {args.project}/{args.name}\n")

    train_kwargs = dict(
        data      = str(data_yaml),
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        workers   = args.workers,
        project   = str(args.project),
        name      = args.name,
        patience  = args.patience,
        exist_ok  = True,
        verbose   = True,
    )
    if args.device != "":
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    # Copy best checkpoint to models/best.pt
    run_dir  = Path(args.project) / args.name
    best_src = run_dir / "weights" / "best.pt"

    if best_src.exists():
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, out_path)
        print(f"\n  Best checkpoint → {out_path.resolve()}")
    else:
        print(f"\n  WARNING: best.pt not found at {best_src}")
        print(f"  Check {run_dir}/weights/ for available checkpoints.")

    # Print final metrics summary
    print("\n  Training complete.")
    try:
        metrics = results.results_dict
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                    "metrics/precision(B)", "metrics/recall(B)"):
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
