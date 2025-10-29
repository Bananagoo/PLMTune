#!/bin/bash -l
#SBATCH --job-name=plmtune-interpret
#SBATCH --output=logs/interpret_%j.out
#SBATCH --error=logs/interpret_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --export=ALL

echo "=== [PLMTune] Interpretability Environment Setup ==="
echo "Running on $(hostname)"
echo "Time: $(date)"

module load python/3.11.3_torch_gpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune || exit 1
export PYTHONPATH=$(pwd)/src
export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME" logs
echo "HF cache directory: $HF_HOME"

echo "Installing dependencies from requirements.txt ..."
grep -v -E "^(torch|torchvision)" requirements.txt > requirements_hpc.txt
pip install --user -r requirements_hpc.txt
pip cache purge -q

# Parameterized defaults
MODE=${MODE:-all}                    # attention|grad|ig|sae|all
CSV=${CSV:-data/processed/val.csv}
CKPT=${CKPT:-best.pt}
MODEL=${MODEL:-t6}
BATCH_SIZE=${BATCH_SIZE:-8}
N_SAMPLES=${N_SAMPLES:-64}
PROJECT=${PROJECT:-idr-vep-esm2}
RUN_PREFIX=${RUN_PREFIX:-interp}
OFFLINE=${OFFLINE:-0}

# SAE params
CODE_DIM=${CODE_DIM:-256}
SAE_EPOCHS=${SAE_EPOCHS:-10}
SAE_BATCH=${SAE_BATCH:-128}
SAE_LR=${SAE_LR:-1e-3}
L1=${L1:-1e-3}
IG_STEPS=${IG_STEPS:-32}

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

run_attention() {
  echo "--- Attention Rollout ---"
  python scripts/interpret.py \
    --mode attention \
    --csv "$CSV" \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --n_samples "$N_SAMPLES" \
    --out_dir "outputs/interpret/${RUN_PREFIX}-attention-${SLURM_JOB_ID:-local}" \
    --project "$PROJECT" \
    --run_name "${RUN_PREFIX}-attention"
}

run_grad() {
  echo "--- Gradient Saliency ---"
  python scripts/interpret.py \
    --mode grad \
    --ckpt "$CKPT" \
    --csv "$CSV" \
    --batch_size "$BATCH_SIZE" \
    --n_samples "$N_SAMPLES" \
    --out_dir "outputs/interpret/${RUN_PREFIX}-grad-${SLURM_JOB_ID:-local}" \
    --project "$PROJECT" \
    --run_name "${RUN_PREFIX}-grad"
}

run_ig() {
  echo "--- Integrated Gradients ---"
  python scripts/interpret.py \
    --mode ig \
    --ckpt "$CKPT" \
    --csv "$CSV" \
    --ig_steps "$IG_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --n_samples "$N_SAMPLES" \
    --out_dir "outputs/interpret/${RUN_PREFIX}-ig-${SLURM_JOB_ID:-local}" \
    --project "$PROJECT" \
    --run_name "${RUN_PREFIX}-ig"
}

run_sae() {
  echo "--- Sparse Autoencoder Concepts ---"
  python scripts/interpret.py \
    --mode sae \
    --ckpt "$CKPT" \
    --csv "$CSV" \
    --batch_size "$BATCH_SIZE" \
    --n_samples "$N_SAMPLES" \
    --code_dim "$CODE_DIM" \
    --sae_epochs "$SAE_EPOCHS" \
    --sae_batch "$SAE_BATCH" \
    --sae_lr "$SAE_LR" \
    --l1 "$L1" \
    --out_dir "outputs/interpret/${RUN_PREFIX}-sae-${SLURM_JOB_ID:-local}" \
    --project "$PROJECT" \
    --run_name "${RUN_PREFIX}-sae"
}

echo "=== [PLMTune] Running Interpretability (${MODE}) ==="
case "$MODE" in
  attention) run_attention ;;
  grad)      run_grad ;;
  ig)        run_ig ;;
  sae)       run_sae ;;
  all)
    run_attention
    run_grad
    run_ig
    run_sae
    ;;
  *) echo "Unknown MODE: $MODE"; exit 1 ;;
esac

echo "=== [PLMTune] Interpretability Done ==="
