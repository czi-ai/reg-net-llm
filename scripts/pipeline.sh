#!/bin/bash

# Prerequisite:
# To run this pipeline, the CellxGene dataset download and parsing from tissue-type to cell-type must already have been completed
# From the "<PATH_TO_REPO>/scGraphLLM/scripts" directory, run the following script... (i.e. cd /hpc/projects/group.califano/GLM/scGraphLLM/scripts)

# EXAMPLE COMMAND (Preprocessing): 
# source pipeline.sh --preprocess --data-dir "/hpc/projects/group.califano/GLM/data" --cxg-dir "/hpc/projects/group.califano/GLM/data/cellxgene/data/cell_type_all" --out-dir "/hpc/projects/group.califano/GLM/data/cellxgene/data/" --rank-by-z --aracne-top-n-hvg "1024" --aracne-path "/hpc/projects/group.califano/GLM/ARACNe3/build/src/app/ARACNe3_app_release" --job-out-dir "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out"
# source pipeline.sh --cache --data-dir "/hpc/projects/group.califano/GLM/data" --out-dir "/hpc/archives/group.califano" --aracne-top-n-hvg "1024" --job-out-dir "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out"

# Initialize variables:
PREPROCESS=false # Preprocess the CellxGene raw cell-type data (true/false)
CACHE=false # Cache the preprocessed CellxGene cell-type data (true/false)
DATA_DIR="" # Directory to create the cache folder(s) - one folder for each ARACNE_TOP_N_HVG configuration
CXG_DIR="" # Path to the CellxGene cell-type dataset directory (i.e. <PATH>/cell_type_all)
OUT_DIR="" # Path to store the processed data. The same <PATH> as cell_type_all (above) is recommended
RANK_BY_Z=false # When creating the expression rank bins, rank by the expression z-score of each gene? or just the raw expression value (true/false)
ARACNE_TOP_N_HVG="" # TOP_N_HVG to preprocess: 1024, 2048, 4096, etc
ARACNE_PATH="" # Path to the compiled ARACNe3 algorithm (i.e. /hpc/projects/group.califano/GLM/ARACNe3/build/src/app/ARACNe3_app_release)
JOB_OUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --preprocess)
      PREPROCESS=true
      shift # Remove --preprocess
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
      RANK_BY_Z=true
      shift # Remove --rank-by-z
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
    --cache)
      CACHE=true
      shift # Remove --cache
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo ""
echo "  ________     ___    ___ ________          ________  ___  ________  _______   ___       ___  ________   _______       "
echo " |\   ____\   |\  \  /  /|\   ____\        |\   __  \|\  \|\   __  \|\  ___ \ |\  \     |\  \|\   ___  \|\  ___ \      "
echo " \ \  \___|   \ \  \/  / | \  \___|        \ \  \|\  \ \  \ \  \|\  \ \   __/|\ \  \    \ \  \ \  \\ \  \ \   __/|     "
echo "  \ \  \       \ \    / / \ \  \  ___       \ \   ____\ \  \ \   ____\ \  \_|/_\ \  \    \ \  \ \  \\ \  \ \  \_|/__   "
echo "   \ \  \____   /     \/   \ \  \|\  \       \ \  \___|\ \  \ \  \___|\ \  \_|\ \ \  \____\ \  \ \  \\ \  \ \  \_|\ \  "
echo "    \ \_______\/  /\   \    \ \_______\       \ \__\    \ \__\ \__\    \ \_______\ \_______\ \__\ \__\\ \__\ \_______\ "
echo "     \|_______/__/ /\ __\    \|_______|        \|__|     \|__|\|__|     \|_______|\|_______|\|__|\|__| \|__|\|_______| "
echo "              |__|/ \|__|                                                                                              "
echo ""   


CACHE_DIR=""
# Directory naming
# HAVE TO MAKE SOME CHANGES FOR JOB_OUT_DIR
if $RANK_BY_Z; then
    OUT_DIR="$OUT_DIR"/complete_data_ranked_z_score
    JOB_OUT_DIR="$JOB_OUT_DIR"/"$ARACNE_TOP_N_HVG"_z_scored # Directory where to store the job .out files
    CACHE_DIR="$DATA_DIR"/cxg_cache_"$ARACNE_TOP_N_HVG"
else
    OUT_DIR="$OUT_DIR"/complete_data_base
    JOB_OUT_DIR="$JOB_OUT_DIR"/"$ARACNE_TOP_N_HVG"_base # Directory where to store the job .out files
    CACHE_DIR="$DATA_DIR"/cxg_cache_"$ARACNE_TOP_N_HVG"_base
fi

# Use the parsed values
echo ""
echo "PREPROCESS: $PREPROCESS"
echo "CACHE: $CACHE"
echo ""
echo "DATA_DIR: $DATA_DIR"
echo "CXG_DIR: $CXG_DIR"
echo "OUT_DIR: $OUT_DIR"
echo "CACHE_DIR: $CACHE_DIR"
echo "RANK_BY_Z: $RANK_BY_Z"
echo "ARACNE_TOP_N_HVG: $ARACNE_TOP_N_HVG"
echo "ARACNE_PATH: $ARACNE_PATH"
echo "JOB_OUT_DIR: $JOB_OUT_DIR"
echo ""

# Preprocess the different cell-lines
if $PREPROCESS; then
  echo ""
  echo "  ____                                             _                          "
  echo " |  _ \ _ __ ___ _ __  _ __ ___   ___ ___  ___ ___(_)_ __   __ _              "
  echo " | |_) | °__/ _ \ °_ \| °__/ _ \ / __/ _ \/ __/ __| | °_ \ / _° |             "
  echo " |  __/| | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |  _   _   _  "
  echo " |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, | (_) (_) (_) "
  echo "                |_|                                        |___/              "
  echo ""

  # source start_slurm_preprocess.sh \
  #     --run-type "initial" \
  #     --data-dir $DATA_DIR \
  #     --cxg-dir $CXG_DIR \
  #     --out-dir $OUT_DIR \
  #     --rank-by-z $RANK_BY_Z \
  #     --aracne-top-n-hvg $ARACNE_TOP_N_HVG \
  #     --aracne-path $ARACNE_PATH \
  #     --job-out-dir $JOB_OUT_DIR
fi

# Cache the successfully preprocessed cells
# Make sure the previous step is complete before initiating this caching step
# Also ensure that cellxgene_gene2index.csv is in the DATA_PATH directory
if $CACHE; then

  echo ""
  echo "   ____           _     _                          "
  echo "  / ___|__ _  ___| |__ (_)_ __   __ _              "
  echo " | |   / _° |/ __| °_ \| | °_ \ / _° |             "
  echo " | |__| (_| | (__| | | | | | | | (_| |  _   _   _  "
  echo "  \____\__,_|\___|_| |_|_|_| |_|\__, | (_) (_) (_) "
  echo "                                |___/              "
  echo ""

  # # Record which cell-types were preprocessed successfully
  # python success.py --dataset_slurm_out_path $JOB_OUT_DIR

  # # Create the csv, mapping each cell-line to a dataset (Seen Graph: SG, or Held Out Graph: HOG)
  # python outdir_gen.py \
  #     --data-path $DATA_DIR \
  #     --cxg-path $OUT_DIR \
  #     --slurm-out-path $JOB_OUT_DIR \
  #     --aracne-top-n-hvg $ARACNE_TOP_N_HVG
      
  # source start_slurm_cache.sh \
  #     --data-dir $DATA_DIR \
  #     --cache-dir $CACHE_DIR \
  #     --aracne-top-n-hvg $ARACNE_TOP_N_HVG \
  #     --job-out-dir $JOB_OUT_DIR
fi