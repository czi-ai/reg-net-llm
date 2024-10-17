#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --output=./slurm_out_%A/array_job_%A_%a.out
#SBATCH -p cpu

# Create the directory for output and error files
mkdir -p ./slurm_out_${SLURM_ARRAY_JOB_ID}

DIRECTORY="$1"
RUN_TYPE="$2"
dir="$3"
N_TOP_HVG="$4"
JOB_ID="$5"

if [ $RUN_TYPE == "initial" ]; then
    CELL_TYPES=($(ls -1 "$DIRECTORY" | sort)) # List of cell-types in the given directory

elif [ $RUN_TYPE == "run_env_fail" ]; then
    CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/env_fail.txt"))

elif [ $RUN_TYPE == "run_timed_out" ]; then
    CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/timed_out.txt"))

elif [ $RUN_TYPE == "run_failed" ]; then
    CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/failed.txt"))

elif [ $RUN_TYPE == "run_unaccounted" ]; then
    CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/unaccounted.txt"))
fi

CELL_TYPE=${CELL_TYPES[$SLURM_ARRAY_TASK_ID-1]}

# echo "Starting processing on ${CELL_TYPES}..."
echo "Starting processing on ${CELL_TYPE}..."

# Execute the preprocessing pipeline on this cell-type
/hpc/projects/group.califano/GLM/scGraphLLM/scripts/preprocess_cellxgene.sh "$DIRECTORY" "$CELL_TYPE" "$N_TOP_HVG"