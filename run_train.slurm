#!/bin/bash
#SBATCH --job-name=nocaps_train
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate /ocean/projects/${PROJECT}/nadayanu/envs/nocaps

# Fail fast on any error
set -e

REPO_ROOT="/ocean/projects/${PROJECT}/nadayanu/nocaps"
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
    --batch       32 \
    --imgsz       640 \
    --device      0 \
    --workers     8 \
    --patience    30 \
    --project     runs/train \
    --name        billiards_v1 \
    --out         models/best.pt

echo ""
echo "=== Done. Best checkpoint: ${REPO_ROOT}/models/best.pt ==="
