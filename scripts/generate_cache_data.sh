#!/bin/bash
#SBATCH --partition=preempted
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --job-name=Generating_LLM_Data

module load mamba
mamba activate scllm

DATA_DIR="" # Path to where the cache data should be stored
CACHE_DIR=""
ARACNE_TOP_N_HVG="" # Term for how many highly variable genes (HVG) to use in dataset processing (ARACNe etc.)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"
      shift # Remove data-dir
      shift # Remove value1
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift
      shift
      ;;
    --aracne-top-n-hvg)
      ARACNE_TOP_N_HVG="$2"
      shift # Remove --aracne-top-n-hvg
      shift # Remove value2
      ;;
  esac
done

python ../scGraphLLM/data.py \
    --aracane-outdir-md "$DATA_DIR"/aracne_"$ARACNE_TOP_N_HVG"_outdir.csv \
    --gene-to-node-file "$DATA_DIR"/cellxgene_gene2index.csv \
    --cache-dir "$CACHE_DIR" \
    --single-index $SLURM_ARRAY_TASK_ID

# python ../scGraphLLM/data.py \
#     --aracane-outdir-md /hpc/projects/group.califano/GLM/data/aracne_2048_outdir.csv \
#     --gene-to-node-file /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv \
#     --cache-dir /hpc/mydata/leo.dupire/data/cellxgene \
#     --single 4