#!/bin/bash
#SBATCH --partition=preempted
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --job-name=Generating_LLM_Data

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --aracane-outdir-md) ARACNE_OUTDIR_MD="$2"; shift 2 ;;
    --gene-to-node-file) GENE_TO_NODE_FILE="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    # --perturbed) PERTURBED="$2"; shift 2 ;;
    # --gene_id) GENE_ID="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

module load mamba
mamba activate scllm

# Execute the caching pipeline on this dataset
python ../scGraphLLM/data.py \
    --aracane-outdir-md $ARACNE_OUTDIR_MD \
    --gene-to-node-file $GENE_TO_NODE_FILE \
    --cache-dir $CACHE_DIR \
    --single-index $SLURM_ARRAY_TASK_ID

 # --perturbed "$PERTURBED" \
#  --gene_id $GENE_ID \

# #-- SBATCH --partition=preempted
# #-- SBATCH --requeue
