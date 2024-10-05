#!/bin/bash

#SBATCH --job-name=Generating_LLM_Data
#SBATCH --output=./slurm_data_out/array_job_%A_%a.out
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1


# Display all variables set by slurm
env | grep "^SLURM" | sort

# Print hostname job executed on.
echo
echo "My hostname is: $(hostname -s)"
echo

module load mamba
mamba activate scllm

python scGraphLLM/data.py --aracane-outdir-md  /hpc/projects/group.califano/GLM/data/aracne_outdir.csv --gene-to-node-file /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv --cache-dir /hpc/projects/group.califano/GLM/data/pilotdata --num-proc 64