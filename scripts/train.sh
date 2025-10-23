#!/bin/bash
#SBATCH --job-name=plmtune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH -G 1

module load python/3.9.21

# Activate your virtual environment
source /path/to/your/writable/folder/PLMTune/plmtune_env/bin/activate

# Move into repo
cd /path/to/your/writable/folder/PLMTune

# Check CUDA visibility
nvidia-smi

# Run training
python scripts/train.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --project idr-vep-esm2