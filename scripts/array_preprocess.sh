#!/bin/bash
#SBATCH --partition=preempted
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G

# Create the directory for output and error files
mkdir -p ./slurm_out_${SLURM_ARRAY_JOB_ID}

RUN_TYPE=""
DATA_DIR=""
CXG_DIR=""
OUT_DIR=""
RANK_BY_Z=""
ARACNE_TOP_N_HVG=""
ARACNE_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-type)
      RUN_TYPE="$2"
      shift # Remove --run-type
      shift # Remove value1
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift # Remove --data-dir
      shift # Remove value8
      ;;
    --cxg-dir)
      CXG_DIR="$2"
      shift # Remove --cxg-dir
      shift # Remove value2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift # Remove --out-dir
      shift # Remove value3
      ;;
    --rank-by-z)
      RANK_BY_Z="$2"
      shift # Remove --rank-by-z
      shift # Remove value4
      ;;
    --aracne-top-n-hvg)
      ARACNE_TOP_N_HVG="$2"
      shift # Remove --aracne-top-n-hvg
      shift # Remove value5
      ;;
    --aracne-path)
      ARACNE_PATH="$2"
      shift # Remove --aracne-path
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ $RUN_TYPE == "initial" ]; then
    CELL_TYPES=($(ls -1 "$CXG_DIR" | sort)) # List of cell-types in the given directory
fi
# elif [ $RUN_TYPE == "run_env_fail" ]; then
#     CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/env_fail.txt"))
# elif [ $RUN_TYPE == "run_timed_out" ]; then
#     CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/timed_out.txt"))
# elif [ $RUN_TYPE == "run_failed" ]; then
#     CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/failed.txt"))
# elif [ $RUN_TYPE == "run_unaccounted" ]; then
#     CELL_TYPES=($(cat "$dir/slurm_out_$JOB_ID/check_out/unaccounted.txt"))
# fi

CELL_TYPE=${CELL_TYPES[$SLURM_ARRAY_TASK_ID-1]}

# echo "Starting processing on ${CELL_TYPES}..."
echo "Starting processing on ${CELL_TYPE}..."

# Execute the preprocessing pipeline on this cell-type
source ./preprocess_cellxgene.sh \
        --cell-type $CELL_TYPE \
        --data-dir $DATA_DIR \
        --cxg-dir $CXG_DIR \
        --out-dir $OUT_DIR \
        --rank-by-z $RANK_BY_Z \
        --aracne-top-n-hvg $ARACNE_TOP_N_HVG \
        --aracne-path $ARACNE_PATH