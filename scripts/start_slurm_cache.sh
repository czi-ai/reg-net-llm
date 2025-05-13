#!/bin/bash

# Initialize variables
ARACNE_OUTDIR_MD="" # Path to aracne_"$ARACNE_TOP_N_HVG"_outdir.csv
GENE_TO_NODE_FILE="" # Path to cellxgene_gene2index.csv
CACHE_DIR="" # Path to where the cache data should be stored
$PERTURBATION_VAR="null" # For perturbation ONLY: column in dataset.obs corresponding to the perturbed gene_id (ENSEMBL symbol notation)
NUM_PROC="1"
JOB_OUT_DIR="" # Where to store all log files from this parallelized caching process

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --aracane-outdir-md) ARACNE_OUTDIR_MD="$2"; shift 2 ;;
    --gene-to-node-file) GENE_TO_NODE_FILE="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --perturbation-var) PERTURBATION_VAR="$2"; shift 2 ;;
    --num-proc) NUM_PROC="$2"; shift 2 ;;
    --job-out-dir) JOB_OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo ""
echo "   ____           _     _                          "
echo "  / ___|__ _  ___| |__ (_)_ __   __ _              "
echo " | |   / _° |/ __| °_ \| | °_ \ / _° |             "
echo " | |__| (_| | (__| | | | | | | | (_| |  _   _   _  "
echo "  \____\__,_|\___|_| |_|_|_| |_|\__, | (_) (_) (_) "
echo "                                |___/              "
echo ""
echo ""
echo -e "aracane-outdir-md:\t $ARACNE_OUTDIR_MD"
echo -e "gene-to-node-file:\t $GENE_TO_NODE_FILE"
echo -e "cache-dir:\t\t $CACHE_DIR"
echo -e "perturbation-var: \t $PERTURBATION_VAR"
echo -e "num-proc: \t\t $NUM_PROC"
echo -e "job-out-dir:\t\t $JOB_OUT_DIR"
echo ""
echo ""

# Get file count - this will be the number of parallel processes in the following sbatch call
FILE_COUNT=$(wc -l $ARACNE_OUTDIR_MD | awk '{print $1}')

# Start the job and parallelize
sbatch --array=1-${FILE_COUNT} --output=${JOB_OUT_DIR}/cache_out/array_job_%A_%a.out ./generate_cache_data.sh \
    --aracane-outdir-md $ARACNE_OUTDIR_MD \
    --gene-to-node-file $GENE_TO_NODE_FILE \
    --cache-dir $CACHE_DIR \
    --perturbation-var $PERTURBATION_VAR \
    --num-proc $NUM_PROC 


##########################################################################################
#################################### EXAMPLE COMMANDS ####################################
##########################################################################################

: <<'END_COMMENT'

# CellxGene
# CellxGene metacell commands
python success.py --dataset_slurm_out_path "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/cxg/4096_meta"

python outdir_gen.py \
      --cxg-path "/hpc/projects/group.califano/GLM/data/_cellxgene/data/cxg_meta_cell_clean" \
      --outfile-path "/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096_outdirs.csv" \
      --slurm-out-path "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/cxg/4096_meta" \
      --aracne-top-n-hvg 4096

# CellxGene caching command
source start_slurm_cache.sh \
    --aracane-outdir-md "/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096_outdirs.csv" \
    --gene-to-node-file "/hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv" \
    --cache-dir "/hpc/projects/group.califano/GLM/data/meta_cell/cxg_4096" \
    --job-out-dir "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/cxg/4096_meta"



# Replogle
python success.py --dataset_slurm_out_path "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/replogle/200k"

python outdir_gen.py \
      --cxg-path "/hpc/projects/group.califano/GLM/data/_replogle/data/replogle_200k_clean" \
      --outfile-path "/hpc/projects/group.califano/GLM/data/single_cell/replogle_outdirs.csv" \
      --slurm-out-path "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/replogle/200k" \
      --aracne-top-n-hvg null

# Replogle caching command
source start_slurm_cache.sh \
    --aracane-outdir-md "/hpc/projects/group.califano/GLM/data/single_cell/replogle_full_outdirs.csv" \
    --gene-to-node-file "/hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv" \
    --cache-dir "/hpc/projects/group.califano/GLM/data/single_cell/replogle_200k" \
    --perturbation-var "gene_id" \
    --job-out-dir "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/replogle/200k"

END_COMMENT