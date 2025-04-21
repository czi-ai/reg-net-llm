#!/bin/bash 

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cell-type) CELL_TYPE="$2"; shift 2 ;;
    --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --aracne-top-n-hvg) ARACNE_TOP_N_HVG="$2"; shift 2 ;;
    --aracne-path) ARACNE_PATH="$2"; shift 2 ;;
    --regulators-path) REGULATORS_PATH="$2"; shift 2 ;;
    --index-vars) INDEX_VARS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --perturbed) PERTURBED="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

preprocess=true
aracne=true
min_total_subnets=50
aracne_dirname=aracne_$ARACNE_TOP_N_HVG
n_threads=4
preprocess_path="../scGraphLLM/preprocess.py"

# activate virtual environment
module load mamba
mamba activate scllm

# Preprocess cell-type
echo "Processing ${CELL_TYPE}..."
start_time=$(date +%s)
if $preprocess; then
    if [[ "$ARACNE_TOP_N_HVG" == "null" ]]; then # consider all genes
        python $preprocess_path \
            --data_path "${RAW_DATA_DIR}/${CELL_TYPE}/partitions" \
            --out_dir "${OUT_DIR}/${CELL_TYPE}" \
            --save_metacells \
            --sample_index_vars $INDEX_VARS \
            --aracne_min_n 250 \
            --aracne_dirname $aracne_dirname \
            --n_bins 250 \
            --dataset $DATASET \
            --perturbed $PERTURBED
    else # Consider ARACNE_TOP_N_HVG genes (no ARACNE_TOP_N_HVG argument passed)
        python $preprocess_path \
            --data_path "${RAW_DATA_DIR}/${CELL_TYPE}/partitions" \
            --out_dir "${OUT_DIR}/${CELL_TYPE}" \
            --save_metacells \
            --sample_index_vars $INDEX_VARS \
            --aracne_min_n 250 \
            --aracne_dirname $aracne_dirname \
            --n_bins 250 \
            --aracne_top_n_hvg $ARACNE_TOP_N_HVG \
            --dataset $DATASET \
            --perturbed $PERTURBED
    fi

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Finished preprocessing ${CELL_TYPE}"
    else
        echo "Failed to preprocess ${CELL_TYPE}"
    fi
fi

echo -e "\n"

if $aracne; then
    # Generate ARACNe subnetworks for each cluster
    counts_files=($(find "$OUT_DIR/$CELL_TYPE/$aracne_dirname/counts/" -name "counts_*.tsv" | sort))
    n_clusters=${#counts_files[@]}

    if [ $n_clusters -eq 0 ]; then
        echo "No clusters detected for ${CELL_TYPE}. ARACNe step incomplete."
    else
        n_subnets_per_cluster=$(((min_total_subnets + $n_clusters) / $n_clusters))
        n_total_subnets=$((n_clusters * n_subnets_per_cluster)) # make this ~50-100
        echo "Generating ${n_subnets_per_cluster} ARACNe subnetworks for ${n_clusters} clusters...${n_total_subnets} total subnetworks..."
        for file_path in "${counts_files[@]}"
        do  
            index=$(basename "$file_path" | sed 's/counts_\([0-9]*\)\.tsv/\1/')
            echo "Running ARACNe for ${file_path} (Cluster: $index)" 
            
            $ARACNE_PATH \
                -e $file_path \
                -r $REGULATORS_PATH \
                -o $OUT_DIR/$CELL_TYPE/$aracne_dirname \
                -x $n_subnets_per_cluster \
                -FDR \
                --runid $index \
                --alpha 0.05 \
                --noConsolidate \
                --seed 12345 \
                --threads $n_threads
        done

        echo "Consolidating ARACNe subnetworks for ${n_clusters} clusters of cell type: ${CELL_TYPE} "
        $ARACNE_PATH \
            -e $OUT_DIR/$CELL_TYPE/$aracne_dirname/counts/counts.tsv \
            -r $REGULATORS_PATH \
            -o $OUT_DIR/$CELL_TYPE/$aracne_dirname \
            -x $n_total_subnets \
            -FDR \
            --subsample 1.00 \
            --alpha 0.05 \
            --consolidate \
            --seed 12345 \
            --threads $n_threads

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