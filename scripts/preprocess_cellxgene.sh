#!/bin/bash 

# CXG_DIR="/hpc/projects/group.califano/GLM/data/cellxgene/data/cell_type_all"
# CELL_TYPE="macrophage" # "astrocyte_of_the_cerebral_cortex"
# ARACNE_TOP_N_HVG=1024

CELL_TYPE=""
DATA_DIR=""
CXG_DIR=""
OUT_DIR=""
RANK_BY_Z=""
ARACNE_TOP_N_HVG=""
ARACNE_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cell-type)
      CELL_TYPE="$2"
      shift # Remove --cell-type
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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

PREPROCESS=true
RUN_ARACNE=true
MIN_TOTAL_SUBNETS=50
ARACNE_DIRNAME=aracne_$ARACNE_TOP_N_HVG
N_THREADS=4

# activate virtual environment
module load mamba
mamba activate scllm

# Base paths
regulators_path="$DATA_DIR"/regulators.txt
preprocess_path="../scGraphLLM/preprocess.py"
aracne=$ARACNE_PATH

# Preprocess cell-type
echo "Processing ${CELL_TYPE}..."

start_time=$(date +%s)
if $PREPROCESS; then
    if $RANK_BY_Z_SCORE; then
        OUT_DIR="$OUT_DIR"
        python $preprocess_path \
            --data_path "${CXG_DIR}/${CELL_TYPE}/partitions" \
            --out_dir "${OUT_DIR}/${CELL_TYPE}" \
            --save_metacells \
            --sample_index_vars dataset_id donor_id tissue \
            --aracne_min_n 250 \
            --aracne_dirname $ARACNE_DIRNAME \
            --n_bins 250 \
            --aracne_top_n_hvg $ARACNE_TOP_N_HVG \
            --rank_by_z_score
    else
        python $preprocess_path\
            --data_path "${CXG_DIR}/${CELL_TYPE}/partitions" \
            --out_dir "${OUT_DIR}/${CELL_TYPE}" \
            --save_metacells \
            --sample_index_vars dataset_id donor_id tissue \
            --aracne_min_n 250 \
            --aracne_dirname $ARACNE_DIRNAME \
            --n_bins 250 \
            --aracne_top_n_hvg $ARACNE_TOP_N_HVG
    fi

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Finished preprocessing ${CELL_TYPE}"
    else
        echo "Failed to preprocess ${CELL_TYPE}"
    fi
fi

echo -e "\n"

if $RUN_ARACNE; then
    # Generate ARACNe subnetworks for each cluster
    counts_files=($(find "$OUT_DIR/$CELL_TYPE/$ARACNE_DIRNAME/counts/" -name "counts_*.tsv" | sort))
    n_clusters=${#counts_files[@]}

    if [ $n_clusters -eq 0 ]; then
        echo "No clusters detected for ${CELL_TYPE}. ARACNe step incomplete."
    else
        n_subnets_per_cluster=$(((MIN_TOTAL_SUBNETS + $n_clusters) / $n_clusters))
        n_total_subnets=$((n_clusters * n_subnets_per_cluster)) # make this ~50-100
        echo "Generating ${n_subnets_per_cluster} ARACNe subnetworks for ${n_clusters} clusters...${n_total_subnets} total subnetworks..."
        for file_path in "${counts_files[@]}"
        do  
            index=$(basename "$file_path" | sed 's/counts_\([0-9]*\)\.tsv/\1/')
            echo "Running ARACNe for ${file_path} (Cluster: $index)" 
            
            $aracne \
                -e $file_path \
                -r $regulators_path \
                -o $OUT_DIR/$CELL_TYPE/$ARACNE_DIRNAME \
                -x $n_subnets_per_cluster \
                -FDR \
                --runid $index \
                --alpha 0.05 \
                --noConsolidate \
                --seed 12345 \
                --threads $N_THREADS
        done

        echo "Consolidating ARACNe subnetworks for ${n_clusters} clusters of cell type: ${CELL_TYPE} "
        $aracne \
            -e $OUT_DIR/$CELL_TYPE/$ARACNE_DIRNAME/counts/counts.tsv \
            -r $regulators_path \
            -o $OUT_DIR/$CELL_TYPE/$ARACNE_DIRNAME \
            -x $n_total_subnets \
            -FDR \
            --subsample 1.00 \
            --alpha 0.05 \
            --consolidate \
            --seed 12345 \
            --threads $N_THREADS

        # Check if the ARACNe command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully ran ARACNe for ${CELL_TYPE}"
        else
            echo "Failed to run ARACNe for ${CELL_TYPE}"
        fi
    fi
fi

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Time taken to process ${CELL_TYPE}: ${elapsed_time} seconds"

mamba deactivate