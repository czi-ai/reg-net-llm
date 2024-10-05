#!/bin/bash

RUN_TYPE=$1
JOB_ID=$2

cd /hpc/projects/group.califano/GLM/scGraphLLM/scripts

# Create SLURM output directory
dir="/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out"

if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
    echo "Directory created: $dir"
fi

cd "$dir"

# Directory containing all cell-type directories
DIRECTORY="/hpc/projects/group.califano/GLM/data/cellxgene/data/cell_type_all"

if [ $RUN_TYPE == "initial" ]; then
    # Count the number of cell-type directories
    FILE_COUNT=$(ls -1 "$DIRECTORY" | wc -l)

elif [ $RUN_TYPE == "run_env_fail" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/env_fail.txt")

elif [ $RUN_TYPE == "run_timed_out" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/timed_out.txt")

elif [ $RUN_TYPE == "run_failed" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/failed.txt")

elif [ $RUN_TYPE == "run_unaccounted" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/unaccounted.txt")
fi

# Start the job and parallelize by the number of cell-types
sbatch --array=1-${FILE_COUNT} /hpc/projects/group.califano/GLM/scGraphLLM/scripts/array_preprocess.sh "$DIRECTORY" "$RUN_TYPE" "$dir" "$JOB_ID"
cd ..