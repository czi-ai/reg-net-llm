#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=Generating_LLM_Data

# Usually 512G for 1-2 days

# As a defaults
NUM_PROC=1
PARTITION=0

# If this is a partitioned job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    PARTITION=$SLURM_ARRAY_TASK_ID
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --aracane-outdir-md) ARACNE_OUTDIR_MD="$2"; shift 2 ;;
    --gene-to-node-file) GENE_TO_NODE_FILE="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --perturbation-var) PERTURBATION_VAR="$2"; shift 2 ;;
    --num-proc) NUM_PROC="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

module load mamba
mamba activate scllm

if [[ -n "$NUM_PROC" && "${NUM_PROC,,}" != "1" ]]; then
  SINGLE_INDEX_ARG="--single-index 0"
else
  SINGLE_INDEX_ARG="--single-index $((SLURM_ARRAY_TASK_ID - 1))"
fi

# Execute the caching pipeline on this dataset
python ../scGraphLLM/data.py \
    --aracane-outdir-md $ARACNE_OUTDIR_MD \
    --gene-to-node-file $GENE_TO_NODE_FILE \
    --cache-dir $CACHE_DIR \
    --perturbation-var $PERTURBATION_VAR \
    --num-proc $NUM_PROC \
    --partition $PARTITION \
    $SINGLE_INDEX_ARG
