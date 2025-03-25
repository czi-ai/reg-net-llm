#!/bin/bash 
#SBATCH --partition gpu
#SBATCH --nodes 1 
#SBATCH --gpus 8
#SBATCH --cpus-per-gpu 8
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu 8G
#SBATCH --nodelist gpu-d-[1-2],gpu-f-[1-6],gpu-h-[1-6,8]
#SBATCH --job-name graph_kernel_attn_4096_multigpu_prodrun

srun /hpc/mydata/vinay.swamy/miniforge3/envs/scllm/bin/python scGraphLLM/run_training.py \
    --config graph_kernel_attn_4096 \
    --mode train \
    --name graph_kernel_attn_4096_multigpu_prodrun
