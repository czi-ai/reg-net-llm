#!/bin/bash
#SBATCH -p cpu
#SBATCH --account=pmg
#SBATCH --output=./slurm_out/array_job_%A_%a.out

mkdir slurm_out
cd slurm_out

DIRECTORY="/burg/pmg/collab/scGraphLLM/data/cellxgene/cell_type_005"

CELL_TYPES=($(ls -1 "$DIRECTORY" | sort))
cell_type="${CELL_TYPES[$((${SLURM_ARRAY_TASK_ID}-1))]}"

echo "Preprocessing ${cell_type}"

./preprocess_cellxgene.sh ${cell_type}