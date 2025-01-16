#!/bin/bash

DATA_DIR=""
CACHE_DIR=""
ARACNE_TOP_N_HVG=""
JOB_OUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"
      shift # Remove --data-dir
      shift
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift # Remove --cache-dir
      shift
      ;;
    --aracne-top-n-hvg)
      ARACNE_TOP_N_HVG="$2"
      shift # Remove --aracne-top-n-hvg
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

FILE_COUNT=$(wc -l "$DATA_DIR"/aracne_"$ARACNE_TOP_N_HVG"_outdir.csv | awk '{print $1}')
# Start the job and parallelize
sbatch --array=1-${FILE_COUNT} --output=${JOB_OUT_DIR}/cache_out/array_job_%A_%a.out ./generate_cache_data.sh \
    --data-dir $DATA_DIR \
    --cache-dir $CACHE_DIR \
    --aracne-top-n-hvg $ARACNE_TOP_N_HVG