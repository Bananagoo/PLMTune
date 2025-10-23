#!/bin/bash -l
#SBATCH --job-name=plmtune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

echo "=== [PLMTune] Environment Setup ==="
echo "Running on $(hostname)"
echo "Time: $(date)"

# Load your Python + CUDA environment
module load python/3.11.3_torch_gpu

# Use your large writable scratch directory for pip temp/cache
export TMPDIR=/hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/pip_tmp
mkdir -p $TMPDIR

# Move to repo
cd /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune || exit 1

# Add project root to Python path
export PYTHONPATH=$(pwd)/src

# Create logs folder if missing
mkdir -p logs

echo "Installing dependencies from requirements.txt ..."
grep -v -E "^(torch|torchvision)" requirements.txt > requirements_hpc.txt

# Install everything else
pip install --user --no-deps -r requirements_hpc.txt

# Optional cleanup
pip cache purge -q

echo "=== [PLMTune] Environment setup complete! ==="
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Run your training script
python scripts/train.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --project idr-vep-esm2