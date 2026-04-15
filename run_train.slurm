#!/bin/bash
#SBATCH --job-name=nocaps_train
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250224p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate /ocean/projects/cis250224p/nadayanu/envs/nocaps

# Fix PyTorch/CUDA mismatch: installed build requires CUDA 13 but nodes have 12.6
# This is a no-op if the right version is already present.
pip install -q --upgrade "torch==2.5.1" torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Fail fast on any error
set -e

REPO_ROOT="/ocean/projects/cis250224p/nadayanu/nocaps"
cd "${REPO_ROOT}"

mkdir -p logs models

# ── Step 1: Build base dataset from manual annotations ─────────────────────────
echo "=== Step 1: prepare_yolo_data.py ==="
python scripts/prepare_yolo_data.py \
    --dataset-dir billiards_dataset \
    --out         yolo_dataset

# ── Step 2: Track-propagation autolabeling ─────────────────────────────────────
echo ""
echo "=== Step 2: autolabel.py ==="
python scripts/autolabel.py \
    --dataset-dir billiards_dataset \
    --out         yolo_dataset \
    --felt        blue \
    --stride      2 \
    --min-tracks  4 \
    --seed        42

# ── Step 3: Train YOLOv8-small ─────────────────────────────────────────────────
echo ""
echo "=== Step 3: train_yolo.py ==="
python scripts/train_yolo.py \
    --data        yolo_dataset/data.yaml \
    --model-size  s \
    --epochs      150 \
    --batch       16 \
    --imgsz       640 \
    --workers     4 \
    --patience    30 \
    --project     /ocean/projects/cis250224p/nadayanu/nocaps/runs/train \
    --name        billiards_v1 \
    --out         models/best.pt \
    --resume

echo ""
echo "=== Done. Best checkpoint: ${REPO_ROOT}/models/best.pt ==="
