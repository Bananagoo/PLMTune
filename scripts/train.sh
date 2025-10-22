#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH -G 1
#SBATCH --mem=50G

module load python/3.13.1

python train.py