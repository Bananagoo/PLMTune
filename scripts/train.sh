#!/bin/bash
#SBATCH --job-name=plmtune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH -G 1

# Activate your virtual environment
source /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/plmtune_env/bin/activate

# Move into your project folder
cd /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune

# Check CUDA visibility
nvidia-smi

# Run training
python scripts/train.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --project idr-vep-esm2