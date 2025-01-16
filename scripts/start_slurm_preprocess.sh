#!/bin/bash

RUN_TYPE=""
DATA_DIR=""
CXG_DIR=""
OUT_DIR=""
RANK_BY_Z=""
ARACNE_TOP_N_HVG=""
ARACNE_PATH=""
JOB_OUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-type)
      RUN_TYPE="$2"
      shift # Remove --run-type
      shift
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift # Remove --data-dir
      shift
      ;;
    --cxg-dir)
      CXG_DIR="$2"
      shift # Remove --cxg-dir
      shift
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift # Remove --out-dir
      shift
      ;;
    --rank-by-z)
      RANK_BY_Z="$2"
      shift # Remove --rank-by-z
      shift
      ;;
    --aracne-top-n-hvg)
      ARACNE_TOP_N_HVG="$2"
      shift # Remove --aracne-top-n-hvg
      shift
      ;;
    --aracne-path)
      ARACNE_PATH="$2"
      shift # Remove --aracne-path
      shift
      ;;
    --job-out-dir)
      JOB_OUT_DIR="$2"
      shift # Remove --job-out-dir
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Determine action based on requested run-type
if [ $RUN_TYPE == "initial" ]; then
    # Count the number of cell-type directories
    FILE_COUNT=$(ls -1 "$CXG_DIR" | wc -l)
fi

# elif [ $RUN_TYPE == "run_env_fail" ]; then
#     FILE_COUNT=$(wc -l < "$dir/slurm_out_$JOB_ID/check_out/env_fail.txt")
# elif [ $RUN_TYPE == "run_timed_out" ]; then
#     FILE_COUNT=$(wc -l < "$dir/slurm_out_$JOB_ID/check_out/timed_out.txt")
# elif [ $RUN_TYPE == "run_failed" ]; then
#     FILE_COUNT=$(wc -l < "$dir/slurm_out_$JOB_ID/check_out/failed.txt")
# elif [ $RUN_TYPE == "run_unaccounted" ]; then
#     FILE_COUNT=$(wc -l < "$dir/slurm_out_$JOB_ID/check_out/unaccounted.txt")
# fi

# Create SLURM output directory for this run
if [ ! -d "$JOB_OUT_DIR" ]; then
    mkdir -p "$JOB_OUT_DIR"
fi

# Start the job and parallelize by the number of cell-types
sbatch --array=1-${FILE_COUNT} --output=${JOB_OUT_DIR}/slurm_out_%A/array_job_%A_%a.out ./array_preprocess.sh \
    --run-type $RUN_TYPE \
    --data-dir $DATA_DIR \
    --cxg-dir $CXG_DIR \
    --out-dir $OUT_DIR \
    --rank-by-z $RANK_BY_Z \
    --aracne-top-n-hvg $ARACNE_TOP_N_HVG \
    --aracne-path $ARACNE_PATH 