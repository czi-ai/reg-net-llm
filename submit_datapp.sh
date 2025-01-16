#!/bin/bash 
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --account=pmg
#SBATCH --job-name=build_data_scllm
#SBATCH --nodelist m004
echo $SLURMD_NODENAME
/pmglocal/$USER/mambaforge/envs/scllm/bin/python scGraphLLM/data.py "$@"
