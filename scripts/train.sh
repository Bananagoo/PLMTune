#!/bin/bash -l
#SBATCH --job-name=plmtune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH -G 1
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

# Parameterized defaults (override via env vars before sbatch)
MODEL=${MODEL:-t6}                 # t6 (8M), t12 (35M), t33 (650M) or full HF id
FINETUNE=${FINETUNE:-1}            # 1=fine-tune ESM, 0=freeze ESM
ESM_LR_MULT=${ESM_LR_MULT:-0.1}    # ESM LR multiplier vs head LR when fine-tuning
BATCH_SIZE=${BATCH_SIZE:-8}
SAVE_EVERY=${SAVE_EVERY:-2}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
PROJECT=${PROJECT:-idr-vep-esm2}
RUN_NAME=${RUN_NAME:-small-ft}
TRAIN_CSV=${TRAIN_CSV:-data/processed/train.csv}
VAL_CSV=${VAL_CSV:-data/processed/val.csv}

if [ "$FINETUNE" = "1" ]; then
  FT_ARGS="--finetune_esm --unfreeze_esm --esm_lr_mult ${ESM_LR_MULT}"
else
  FT_ARGS="--freeze_esm"
fi

python scripts/train.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --model "$MODEL" \
  $FT_ARGS \
  --batch_size "$BATCH_SIZE" --save_every "$SAVE_EVERY" --max_grad_norm "$MAX_GRAD_NORM" \
  --project "$PROJECT" \
  --run_name "$RUN_NAME"
