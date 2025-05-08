#!/bin/bash

#SBATCH --job-name=SCAN
#SBATCH --partition=preempted
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=./out/my_cpu_job_%j.out

DATASET="$1"
CACHE_DIR="$2"

module load mamba
mamba activate cellxgene

cd /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted

python corrupt.py "$DATASET" "$CACHE_DIR" "$SLURM_ARRAY_TASK_ID"

# sbatch --array=1-100 /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/scan.sh "train" "/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096"
# sbatch --array=1-100 /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/scan.sh "train" "/hpc/projects/group.califano/GLM/data/single_cell/replogle_200k/perturbed"
# sbatch --array=1-100 /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/scan.sh "val" "/hpc/projects/group.califano/GLM/data/single_cell/replogle_200k/perturbed"
# sbatch --array=1-100 /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/scan.sh "test" "/hpc/projects/group.califano/GLM/data/single_cell/replogle_200k/perturbed"