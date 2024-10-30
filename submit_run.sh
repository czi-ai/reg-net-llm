#!/bin/bash

#SBATCH --job-name=Transformer_TRAIN
#SBATCH --output=./slurm_train_out/array_job_%A_%a.out
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -p gpu

# Display all variables set by slurm
env | grep "^SLURM" | sort

# Print hostname job executed on.
echo
echo "My hostname is: $(hostname -s)"
echo

RUN_NAME=$1

module load mamba
mamba activate scllm_2

python /home/mingxuan.zhang/scGraphLLM/scGraphLLM/run_training.py --config="pilot_graph_pe_run_config" --mode="train" --name="$RUN_NAME"