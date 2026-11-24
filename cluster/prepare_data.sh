#!/bin/bash -l
#SBATCH --job-name=prep_data
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

# Builds dataset cache in outputs_cache from the MOSES csv files in the $MOSES_DIR .
# $MOSES_DIR is mounted as read-only (ro) so IT HAS TO BE THERE and contain
# the files train.csv, test.csv and test_scaffolds.csv in the expected format.
# To make a download of the data instead, set $MOSES_DIR to some writeable dir, 
# e.g. $PWD/moses_data from the repo root (and remove the :ro from the --bind)-
# usage (from the repo root):
#   sbatch --account=<account> cluster/prepare_data.sh [--dataset moses] \ 
#       [--aromatic] (for aromatic bond categotry)

set -e

export DATASETS_DATASET_DIR=/home/space/datasets/qm9_pyg
export MOSES_DIR=${MOSES_DIR:-/home/space/datasets/moses}

apptainer run --nv \
    --bind "$DATASETS_DATASET_DIR:$DATASETS_DATASET_DIR:ro" \
    --bind "$MOSES_DIR:$MOSES_DIR:ro" \
    "${CONTAINER:-$HOME/pml.sif}" \
    python -u -m digress.cli.prepare_data --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
