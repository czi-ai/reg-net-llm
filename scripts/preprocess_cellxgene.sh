#!/bin/bash 
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --nodelist=m012
#SBATCH --account pmg

# activate virtual environment
source activate /pmglocal/$USER/mambaforge/envs/scllm

# Define the list of cell types
cell_types=("photoreceptor_cell")
# cell_types=("type_i_nk_t_cell" "mast_cell" "skin_fibroblast")

# Base paths
data_base_path="/burg/pmg/collab/scGraphLLM/data/cellxgene/cell_type_005"
out_base_path="/burg/pmg/users/$USER/data/cellxgene/data/cell_type_examples"
script_path="/burg/pmg/users/$USER/scGraphLLM/scGraphLLM/preprocess.py"

# Iterate through each cell type
for cell_type in "${cell_types[@]}"
do  
    echo "Processing ${cell_type}..."
    python "$script_path" \
        --data_path "${data_base_path}/${cell_type}/partitions" \
        --out_dir "${out_base_path}/${cell_type}" \
        --save_metacells \
        --produce_figures

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${cell_type}"
    else
        echo "Failed to process ${cell_type}"
    fi
done
