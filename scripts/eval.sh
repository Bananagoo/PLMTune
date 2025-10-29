#!/bin/bash -l
#SBATCH --job-name=plmtune-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --export=ALL

echo "=== [PLMTune] Eval Environment Setup ==="
echo "Running on $(hostname)"
echo "Time: $(date)"

module load python/3.11.3_torch_gpu

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Use a local writable cache next to repo (consistent with train.sh)
cd /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune || exit 1
export PYTHONPATH=$(pwd)/src
export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME" logs
echo "HF cache directory: $HF_HOME"

# Parameterized defaults (override via env before sbatch)
CKPT=${CKPT:-best.pt}
TEST_CSV=${TEST_CSV:-data/processed/test.csv}
BATCH_SIZE=${BATCH_SIZE:-8}
PROJECT=${PROJECT:-idr-vep-esm2}
RUN_NAME=${RUN_NAME:-eval}
OFFLINE=${OFFLINE:-0}

echo "Installing dependencies from requirements.txt ..."
grep -v -E "^(torch|torchvision)" requirements.txt > requirements_hpc.txt
pip install --user -r requirements_hpc.txt
pip cache purge -q

if [ "$OFFLINE" = "1" ]; then
  export TRANSFORMERS_OFFLINE=1
  echo "Transformers offline mode enabled"
fi

# Preflight
python - << 'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
try:
    import transformers
    from transformers import EsmModel
    print('Transformers:', transformers.__version__, '| EsmModel import OK')
except Exception as e:
    print('Transformers import check failed:', e)
PY

echo "=== [PLMTune] Running Evaluation ==="
python scripts/eval.py \
  --ckpt "$CKPT" \
  --test_csv "$TEST_CSV" \
  --batch_size "$BATCH_SIZE" \
  --project "$PROJECT" \
  --run_name "$RUN_NAME"

echo "=== [PLMTune] Eval Done ==="

