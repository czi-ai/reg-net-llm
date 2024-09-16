#!/bin/bash

RUN_TYPE=$1
JOB_ID=$2

cd /burg/pmg/users/$USER/scGraphLLM/scripts

# Create SLURM output directory
dir="/burg/pmg/users/$USER/scGraphLLM/scripts/slurm_out"

if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
    echo "Directory created: $dir"
fi

cd "$dir"

# Directory containing all cell-type directories
DIRECTORY="/burg/pmg/users/rc3686/data/cellxgene/data/cell_type_all"

if [ $RUN_TYPE == "initial" ]; then
    # Count the number of cell-type directories
    FILE_COUNT=$(ls -1 "$DIRECTORY" | wc -l)

elif [ $RUN_TYPE == "run_env_fail" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/env_fail_$JOB_ID.txt")

elif [ $RUN_TYPE == "run_timed_out" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/timed_out_$JOB_ID.txt")

elif [ $RUN_TYPE == "run_failed" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/failed_$JOB_ID.txt")

elif [ $RUN_TYPE == "run_unaccounted" ]; then
    FILE_COUNT=$(wc -l < "$dir/check_out_$JOB_ID/unaccounted_$JOB_ID.txt")
fi

# Start the job and parallelize by the number of cell-types
sbatch --array=1-${FILE_COUNT} /burg/pmg/users/ld3154/scGraphLLM/scripts/array_preprocess.sh "$DIRECTORY" "$RUN_TYPE" "$dir" "$JOB_ID"
cd ..