#!/bin/bash -l
#SBATCH --job-name=plmtune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH -G 1

echo "=== Starting job on $(hostname) ==="
echo "Python path check:"
which /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/plmtune_env/bin/python

# Move to repo
cd /hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune || exit 1

# Run directly with your venv’s Python
/hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/plmtune_env/bin/python -c "import torch; print('Torch version:', torch.__version__)"
/hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/plmtune_env/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

/hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/plmtune_env/bin/python scripts/train.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --project idr-vep-esm2