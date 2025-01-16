#!/bin/bash

#SBATCH --job-name=scan-SG
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --output=my_cpu_job_%j.out

CACHE_DIR="$1"

module load mamba
mamba activate cellxgene

python corrupt_SG.py "$CACHE_DIR"