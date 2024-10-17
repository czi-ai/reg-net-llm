#!/bin/bash 

# DIRECTORY="/hpc/projects/group.califano/GLM/data/cellxgene/data/cell_type_all"
# CELL_TYPE="a2_amacrine_cell"

DIRECTORY=$1
CELL_TYPE=$2
ARACNE_TOP_N_HVG=$3

PREPROCESS=true
RUN_ARACNE=true
MIN_TOTAL_SUBNETS=50
ARACNE_DIRNAME=aracne_$ARACNE_TOP_N_HVG
N_THREADS=4


# activate virtual environment
module load mamba
mamba activate scllm

# Base paths
data_base_path=$DIRECTORY
out_base_path="/hpc/projects/group.califano/GLM/data/cellxgene/data/complete_data"
regulators_path="/hpc/projects/group.califano/GLM/data/regulators.txt"
preprocess_path="/hpc/projects/group.califano/GLM/scGraphLLM/scGraphLLM/preprocess.py"
aracne="/hpc/projects/group.califano/GLM/ARACNe3/build/src/app/ARACNe3_app_release"

# Preprocess cell-type
echo "Processing ${CELL_TYPE}..."

start_time=$(date +%s)
if $PREPROCESS; then
    python $preprocess_path\
        --data_path "${data_base_path}/${CELL_TYPE}/partitions" \
        --out_dir "${out_base_path}/${CELL_TYPE}" \
        --save_metacells \
        --sample_index_vars dataset_id donor_id tissue \
        --aracne_min_n 250 \
        --aracne_top_n_hvg $ARACNE_TOP_N_HVG \
        --aracne_dirname $ARACNE_DIRNAME \
        --n_bins 250

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
    counts_files=($(find "$out_base_path/$CELL_TYPE/$ARACNE_DIRNAME/counts/" -name "counts_*.tsv" | sort))
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
                -o $out_base_path/$CELL_TYPE/$ARACNE_DIRNAME \
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
            -e $out_base_path/$CELL_TYPE/$ARACNE_DIRNAME/counts/counts.tsv \
            -r $regulators_path \
            -o $out_base_path/$CELL_TYPE/$ARACNE_DIRNAME \
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