#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# usage (from the repo root):
#   sbatch --account=<account> --job-name=gt_10k \
#       cluster/train.sh gt_10k simple_graph_transformer [--steps 20000 ...]
#   sbatch --account=<account> --job-name=digress_qm9 \
#       cluster/train.sh digress_qm9 digress --dataset qm9 \ 
#       [--steps 100000 --max-minutes 115 ] new arg --dataset \
#       and --max-minutes for ending before partition end
#   For question of efficiently re-train once new samples become availabel? 
#       --> resubmit with the same run name resumes from `last.pt`
# MOSES dataset: build the cache once with cluster/prepare_data.sh before the training, either on repo root or maybe /home/space/datasets/ ?
set -e

export DATASETS_DATASET_DIR=/home/space/datasets/qm9_pyg
export MOSES_DIR=${MOSES_DIR:-/home/space/datasets/moses}

apptainer run --nv \
    --bind "$DATASETS_DATASET_DIR:$DATASETS_DATASET_DIR:ro" \
    --bind "$MOSES_DIR:$MOSES_DIR:ro" \
    "${CONTAINER:-$HOME/pml.sif}" \
    python -u -m digress.cli.train \
        --run-name "$1" \
        --model "$2" \
        "${@:3}"
