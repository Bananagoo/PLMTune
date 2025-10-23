#!/bin/bash -l
#SBATCH --job-name=plmtune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

echo "=== Starting job on $(hostname) ==="
echo "Loading Python module..."
module load python/3.11.3_torch_gpu

# Move to your repo directory
cd /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune || exit 1

# Add project root to Python path
export PYTHONPATH=$(pwd)

# Check Python and torch availability
which python
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run your training script
python scripts/train.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --project idr-vep-esm2