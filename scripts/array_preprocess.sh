#!/bin/bash
#SBATCH --partition=preempted
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G

# Create the directory for output and error files
mkdir -p ./slurm_out_${SLURM_ARRAY_JOB_ID}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --quantize_metacells) QUANTIZE_METACELLS="$2"; shift 2 ;;
    --aracne-top-n-hvg) ARACNE_TOP_N_HVG="$2"; shift 2 ;;
    --aracne-path) ARACNE_PATH="$2"; shift 2 ;;
    --regulators-path) REGULATORS_PATH="$2"; shift 2 ;;
    --group-by) GROUP_BY="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --perturbed) PERTURBED="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

CELL_TYPES=($(ls -1 "$RAW_DATA_DIR" | sort)) # List of cell-types in the given directory
CELL_TYPE=${CELL_TYPES[$SLURM_ARRAY_TASK_ID-1]} # Get the specific cell-type correspinding to the task ID

# Execute the preprocessing pipeline on this cell-type
echo "Starting processing on ${CELL_TYPE}..."
source ./preprocess_cellxgene.sh \
        --cell-type $CELL_TYPE \
        --raw-data-dir $RAW_DATA_DIR \
        --out-dir $OUT_DIR \
        --quantize_metacells $QUANTIZE_METACELLS \
        --aracne-top-n-hvg $ARACNE_TOP_N_HVG \
        --aracne-path $ARACNE_PATH \
        --regulators-path $REGULATORS_PATH \
        --group-by $GROUP_BY \
        --dataset $DATASET \
        --perturbed $PERTURBED