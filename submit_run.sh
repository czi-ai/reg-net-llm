#!/bin/bash

#SBATCH --job-name=GDTransformer_PRETRAIN
#SBATCH --output=./slurm_train_out/array_job_%A_%a.out
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=h100:8
#SBATCH -p gpu

# Display all variables set by slurm
env | grep "^SLURM" | sort

# Print hostname job executed on.
echo
echo "My hostname is: $(hostname -s)"
echo

RUN_NAME=$1
CONFIG_NAME=$2

module load mamba
mamba activate scllm_2

srun python /home/mingxuan.zhang/scGraphLLM/scGraphLLM/run_training.py --config="$CONFIG_NAME" --mode="train" --name="$RUN_NAME"