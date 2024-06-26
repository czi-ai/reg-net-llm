#!/bin/bash 
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --nodelist=m012
#SBATCH --account pmg

PREPROCESS=false
RUN_ARACNE=true
N_SUBNETS=3

# activate virtual environment
source activate /pmglocal/$USER/mambaforge/envs/scllm

# Define the list of cell types
# cell_types=("type_i_nk_t_cell" "mast_cell" "skin_fibroblast")
cell_types=("neutrophil")
# cell_types=("photoreceptor_cell")

# Base paths
data_base_path="/burg/pmg/collab/scGraphLLM/data/cellxgene/cell_type_005"
out_base_path="/burg/pmg/users/$USER/data/cellxgene/data/cell_type_examples"
regulators_path="/burg/pmg/users/$USER/data/regulators.txt"
alias aracne="/burg/pmg/users/$USER/ARACNe3/build/src/app/ARACNe3_app_release"
alias preprocess="python /burg/pmg/users/$USER/scGraphLLM/scGraphLLM/preprocess.py"

# Iterate through each cell type
for cell_type in "${cell_types[@]}"
do  
    echo "Processing ${cell_type}..."
    start_time=$(date +%s)

    if $PREPROCESS; then
        preprocess \
            --data_path "${data_base_path}/${cell_type}/partitions" \
            --out_dir "${out_base_path}/${cell_type}" \
            --save_metacells \
            --sample_index_vars dataset_id donor_id tissue \
            --aracne_min_n 250

        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully preprocessed ${cell_type}"
        else
            echo "Failed to preprocess ${cell_type}"
        fi
    fi
    
    if $RUN_ARACNE; then
        # Generate ARACNe subnetworks for each cluster
        counts_files=($(find "$out_base_path/$cell_type/aracne/counts/" -name "counts_*.tsv" | sort))
        n_clusters=${#counts_files[@]}
        n_total_subnets=$((n_clusters * N_SUBNETS))
        echo "Generating ${N_SUBNETS} ARACNe subnetworks for ${n_clusters} clusters...${n_total_subnets} total subnetworks..."
        for file_path in "${counts_files[@]}"
        do  
            index=$(basename "$file_path" | sed 's/counts_\([0-9]*\)\.tsv/\1/')
            echo "Running ARACNe for ${file_path} (Index: $index)"
            
            aracne \
                -e $file_path \
                -r $regulators_path \
                -o $out_base_path/$cell_type/aracne \
                -x $N_SUBNETS \
                -FDR \
                --runid $index \
                --subsample 1.00 \
                --alpha 0.05 \
                --noConsolidate \
                --seed 12345
        done

        # Consolidate aracne subnetworks using all counts
        aracne \
            -e $out_base_path/$cell_type/aracne/counts/counts.tsv \
            -r $regulators_path \
            -o $out_base_path/$cell_type/aracne \
            -x $n_total_subnets \
            -FDR \
            --alpha 0.05 \
            --consolidate

        # Check if the ARACNe command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully ran ARACNe for ${cell_type}"
        else
            echo "Failed to run ARACNe for ${cell_type}"
        fi
    fi
    
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Time taken to process ${cell_type}: ${elapsed_time} seconds"
done
