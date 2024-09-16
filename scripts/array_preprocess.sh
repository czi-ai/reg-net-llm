#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --nodelist=m004,m005,m006,m007,m008,m010
#SBATCH --account=pmg
#SBATCH --output=./slurm_out/array_job_%A_%a.out
#SBATCH -p cpu

DIRECTORY="$1"
RUN_TYPE="$2"
dir="$3"
JOB_ID="$4"

if [ $RUN_TYPE == "initial" ]; then
    CELL_TYPES=($(ls -1 "$DIRECTORY" | sort)) # List of cell-types in the given directory

elif [ $RUN_TYPE == "run_env_fail" ]; then
    CELL_TYPES=($(cat "$dir/check_out_$JOB_ID/env_fail_$JOB_ID.txt"))

elif [ $RUN_TYPE == "run_timed_out" ]; then
    CELL_TYPES=($(cat "$dir/check_out_$JOB_ID/timed_out_$JOB_ID.txt"))

elif [ $RUN_TYPE == "run_failed" ]; then
    CELL_TYPES=($(cat "$dir/check_out_$JOB_ID/failed_$JOB_ID.txt"))

elif [ $RUN_TYPE == "run_unaccounted" ]; then
    CELL_TYPES=($(cat "$dir/check_out_$JOB_ID/unaccounted_$JOB_ID.txt"))
fi

cell_type=${CELL_TYPES[$SLURM_ARRAY_TASK_ID-1]}

# echo "Starting processing on ${CELL_TYPES}..."
echo "Starting processing on ${cell_type}..."

# Execute the preprocessing pipeline on this cell-type
/burg/pmg/users/ld3154/scGraphLLM/scripts/preprocess_cellxgene.sh "$DIRECTORY" "$cell_type"
######## SBATC --nodelist=m004,m005,m006,m007,m008,m010