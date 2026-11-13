#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# usage (from the repo root):
#   sbatch --account=<account> --job-name=gt_10k \
#       cluster/train.sh gt_10k simple_graph_transformer [--steps 20000 ...]
set -e

export DATASETS_DATASET_DIR=/home/space/datasets/qm9_pyg

apptainer run --nv \
    --bind "$DATASETS_DATASET_DIR:$DATASETS_DATASET_DIR:ro" \
    "${CONTAINER:-$HOME/pml.sif}" \
    python -u -m digress.cli.train \
        --run-name "$1" \
        --model "$2" \
        "${@:3}"
