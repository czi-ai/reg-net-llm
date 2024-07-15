#!/bin/bash 
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --nodelist=m011
#SBATCH --account pmg

# CELL_TYPE=$1
# ARACNE_TOP_N_HVG=$1

PREPROCESS=true
RUN_ARACNE=true
MIN_TOTAL_SUBNETS=50
ARACNE_TOP_N_HVG=1024
ARACNE_DIRNAME=aracne_$ARACNE_TOP_N_HVG
N_THREADS=4


# activate virtual environment
source activate /pmglocal/$USER/mambaforge/envs/scllm

# Define the list of cell types
# cell_types=("${CELL_TYPE}")
# cell_types=("elicited_macrophage" "mast_cell" "glial_cell" "b_cell" "cd4-positive_alpha-beta_t_cell" "dendritic_cell")
cell_types=("glial_cell" "b_cell" "cd4-positive_alpha-beta_t_cell" "dendritic_cell")
#cell_types=("mast_cell")

# Base paths
data_base_path="/burg/pmg/collab/scGraphLLM/data/cellxgene/cell_type_005"
out_base_path="/burg/pmg/users/$USER/data/cellxgene/data/cell_type_006"
regulators_path="/burg/pmg/users/$USER/data/regulators.txt"
preprocess_path="/burg/pmg/users/$USER/scGraphLLM/scGraphLLM/preprocess.py"
aracne="/burg/pmg/users/$USER/ARACNe3/build/src/app/ARACNe3_app_release"

# Iterate through each cell type
for cell_type in "${cell_types[@]}"
do  
    echo "Processing ${cell_type}..."
    start_time=$(date +%s)

    if $PREPROCESS; then
        python $preprocess_path\
            --data_path "${data_base_path}/${cell_type}/partitions" \
            --out_dir "${out_base_path}/${cell_type}" \
            --steps aracne
            --save_metacells \
            --sample_index_vars dataset_id donor_id tissue \
            --aracne_min_n 250 \
            --aracne_top_n_hvg $ARACNE_TOP_N_HVG \
            --aracne_dirname $ARACNE_DIRNAME \
            --n_bins 100

        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully preprocessed ${cell_type}"
        else
            echo "Failed to preprocess ${cell_type}"
            continue
        fi
    fi
    
    if $RUN_ARACNE; then
        # Generate ARACNe subnetworks for each cluster
        counts_files=($(find "$out_base_path/$cell_type/$ARACNE_DIRNAME/counts/" -name "counts_*.tsv" | sort))
        n_clusters=${#counts_files[@]}

        if [ $n_clusters -eq 0 ]; then
            echo "No clusters detected for ${cell_type}. Skipping ARACNe step."
            continue
        fi

        n_subnets_per_cluster=$(((MIN_TOTAL_SUBNETS + n_clusters) / n_clusters))
        n_total_subnets=$((n_clusters * n_subnets_per_cluster)) # make this ~50-100
        echo "Generating ${n_subnets_per_cluster} ARACNe subnetworks for ${n_clusters} clusters...${n_total_subnets} total subnetworks..."
        for file_path in "${counts_files[@]}"
        do  
            index=$(basename "$file_path" | sed 's/counts_\([0-9]*\)\.tsv/\1/')
            echo "Running ARACNe for ${file_path} (Cluster: $index)" 
            
            $aracne \
                -e $file_path \
                -r $regulators_path \
                -o $out_base_path/$cell_type/$ARACNE_DIRNAME \
                -x $n_subnets_per_cluster \
                -FDR \
                --runid $index \
                --alpha 0.05 \
                --noConsolidate \
                --seed 12345 \
                --threads $N_THREADS
        done

        echo "Consolidating ARACNe subnetworks for ${n_clusters} clusters of cell type: ${cell_type} "
        $aracne \
            -e $out_base_path/$cell_type/$ARACNE_DIRNAME/counts/counts.tsv \
            -r $regulators_path \
            -o $out_base_path/$cell_type/$ARACNE_DIRNAME \
            -x $n_total_subnets \
            -FDR \
            --subsample 1.00 \
            --alpha 0.05 \
            --consolidate \
            --seed 12345 \
            --threads $N_THREADS

        # Check if the ARACNe command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully ran ARACNe for ${cell_type}"
        else
            echo "Failed to run ARACNe for ${cell_type}"
            continue
        fi
    fi
    
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Time taken to process ${cell_type}: ${elapsed_time} seconds"
done
