#!/bin/bash

# Initialize variables
RAW_DATA_DIR="" # Directory in which raw data to be preprocessed is stored. Data is expected to have <RAW_DATA_DIR>/<all cell-type direcotries>/partitions (example path: /hpc/projects/group.califano/GLM/data/cellxgene/data/replogle_raw)
OUT_DIR="" # Where to store the preprocessed data
Z_SCORE_EXPRESSION=true # When creating the expression bins, use the expression z-score expression of each gene? or just the raw expression value (true/false)
ARACNE_TOP_N_HVG="" # Take the top n most highly variable genes (HVGs) for preprocessing: 1024, 2048, 4096, etc
ARACNE_PATH="" # Path to the compiled ARACNe3 algorithm (i.e. /hpc/projects/group.califano/GLM/ARACNe3/build/src/app/ARACNe3_app_release)
REGULATORS_PATH="" # Path to the regulators file (i.e. /hpc/projects/group.califano/GLM/data/regulators.txt)
INDEX_VARS="" # Set of adata.obs columns to consider in grouping steps: "condition" for CellxGene, "gene gene_id transcript" for Replogle
DATASET="" # Dataset name - cellxgene will be treated differently to other datasets
JOB_OUT_DIR="" # Where to store all log files from this parallelization process

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --z-score-expression) Z_SCORE_EXPRESSION=true; shift ;;
    --aracne-top-n-hvg) ARACNE_TOP_N_HVG="$2"; shift 2 ;;
    --aracne-path) ARACNE_PATH="$2"; shift 2 ;;
    --regulators-path) REGULATORS_PATH="$2"; shift 2 ;;
    --index-vars) INDEX_VARS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --job-out-dir) JOB_OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo ""
echo "  ____                                             _                          "
echo " |  _ \ _ __ ___ _ __  _ __ ___   ___ ___  ___ ___(_)_ __   __ _              "
echo " | |_) | °__/ _ \ °_ \| °__/ _ \ / __/ _ \/ __/ __| | °_ \ / _° |             "
echo " |  __/| | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |  _   _   _  "
echo " |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, | (_) (_) (_) "
echo "                |_|                                        |___/              "
echo ""
echo ""
echo -e "raw-data-dir:\t\t $RAW_DATA_DIR"
echo -e "out-dir:\t\t $OUT_DIR"
echo -e "z-score-expression:\t $Z_SCORE_EXPRESSION"
echo -e "aracne-top-n-hvg:\t $ARACNE_TOP_N_HVG"
echo -e "aracne-path:\t\t $ARACNE_PATH"
echo -e "regulators-path:\t $REGULATORS_PATH"
echo -e "index-vars:\t\t $INDEX_VARS"
echo -e "dataset:\t\t $DATASET"
echo -e "job-out-dir:\t\t $JOB_OUT_DIR"
echo ""
echo ""

# Create SLURM output directory for this run
if [ ! -d "$JOB_OUT_DIR" ]; then
    mkdir -p "$JOB_OUT_DIR"
fi

# Get file count - this will be the number of parallel processes in the following sbatch call
FILE_COUNT=$(ls -1 "$RAW_DATA_DIR" | wc -l)

# Start the job and parallelize by the number of cell-types
sbatch --array=1-${FILE_COUNT} --output=${JOB_OUT_DIR}/slurm_out_%A/array_job_%A_%a.out ./array_preprocess.sh \
    --raw-data-dir $RAW_DATA_DIR \
    --out-dir $OUT_DIR \
    --z-score-expression $Z_SCORE_EXPRESSION \
    --aracne-top-n-hvg $ARACNE_TOP_N_HVG \
    --aracne-path $ARACNE_PATH \
    --regulators-path $REGULATORS_PATH \
    --index-vars $INDEX_VARS \
    --dataset $DATASET


##########################################################################################
#################################### EXAMPLE COMMANDS ####################################
##########################################################################################

: <<'END_COMMENT'

# Replogle preprocessing command (as one)
source start_slurm_preprocess.sh \
    --raw-data-dir "/hpc/projects/group.califano/GLM/data/cellxgene/data/replogle_raw" \
    --out-dir "/hpc/projects/group.califano/GLM/data/cellxgene/data/replogle_clean" \
    --z-score-expression \
    --aracne-top-n-hvg "null" \
    --aracne-path "/hpc/projects/group.califano/GLM/ARACNe3/build/src/app/ARACNe3_app_release" \
    --regulators-path "/hpc/projects/group.califano/GLM/data/regulators.txt" \
    --index-vars "gene_id" \
    --dataset "replogle" \
    --job-out-dir "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/replogle/4096"


# Replogle preprocessing command (in 10x200k cells partitions)
source start_slurm_preprocess.sh \
    --raw-data-dir "/hpc/projects/group.califano/GLM/data/cellxgene/data/replogle_raw_partitioned" \
    --out-dir "/hpc/projects/group.califano/GLM/data/cellxgene/data/replogle_clean_partitioned" \
    --z-score-expression \
    --aracne-top-n-hvg "null" \
    --aracne-path "/hpc/projects/group.califano/GLM/ARACNe3/build/src/app/ARACNe3_app_release" \
    --regulators-path "/hpc/projects/group.califano/GLM/data/regulators.txt" \
    --index-vars "gene_id" \
    --dataset "replogle" \
    --job-out-dir "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/replogle/full"

END_COMMENT
