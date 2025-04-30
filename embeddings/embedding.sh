#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load mamba

immune_cell_types=(
  "cd14_monocytes"
  "cd16_monocytes"
  "cd20_b_cells"
  "cd4_t_cells"
  "cd8_t_cells"
  "erythrocytes"
  "monocyte-derived_dendritic_cells"
  "nk_cells"
  "nkt_cells"
)

immune_cell_type_dir="/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type"

# ARACNE_N_HVG=1024
ARACNE_N_HVG=4096

#==============#
#=*= scGPT #=*=#
#==============#

# mamba activate scgpt

# scgpt_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/scgpt_embedding.py"

# for cell_type in "${immune_cell_types[@]}"; do
#     echo "Querying scGPT embeddings for cell type: $cell_type..."
    
#     python $scgpt_embedding_script \
#         --data_dir "$immune_cell_type_dir/$cell_type" \
#         --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scgpt/aracne_$ARACNE_N_HVG" \
#         --model_dir /hpc/mydata/rowan.cassius/scGPT/scGPT_human \
#         --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
#         --scgpt_rootdir /hpc/mydata/rowan.cassius/scGPT \
#         --sample_n_cells 1000
# done

#=====================#
#=*= scFoundation #=*=#
#=====================#

# mamba activate scllm

# scf_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/scfoundation_embedding.py"


# for cell_type in "${immune_cell_types[@]}"; do
#     echo "Querying scGPT embeddings for cell type: $cell_type..."
    
#     python $scf_embedding_script \
#         --data_dir "$immune_cell_type_dir/$cell_type" \
#         --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scfoundation/aracne_$ARACNE_N_HVG" \
#         --model_path /hpc/mydata/rowan.cassius/scFoundation/model/models/models.ckpt \
#         --gene_index_path /hpc/mydata/rowan.cassius/scFoundation/model/OS_scRNA_gene_index.19264.tsv \
#         --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
#         --scf_rootdir /hpc/mydata/rowan.cassius/scFoundation \
#         --sample_n_cells 1000
# done

# ===================#
# =*= scGraphLLM #=*=#
# ===================#

mamba activate scllm

scglm_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/scglm_embedding.py"

for cell_type in "${immune_cell_types[@]}"; do
    echo "Querying scGraphLLM embeddings for cell type: $cell_type..."
    
    python $scglm_embedding_script \
        --data_dir "$immune_cell_type_dir/$cell_type" \
        --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scglm/002_aracne_${ARACNE_N_HVG}" \
        --model_path "/hpc/projects/group.califano/GLM/checkpoints/GK Transformer [noPE, 12Layer] 4096 - PRETRAIN:2025-04-29@19:09:59/checkpoints/epoch=0-step=1000.ckpt.ckpt" \
        --gene_index_path /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv \
        --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
        --sample_n_cells 1000 \
        # --use_masked_edges \
        # --mask_ratio 0.45
done


# ===================#
# =*= Geneformer #=*=#
# ===================#
      
# mamba activate llm

# gf_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/geneformer_embedding.py"

# for cell_type in "${immune_cell_types[@]}"; do
#     echo "Querying scGraphLLM embeddings for cell type: $cell_type..."
    
#     python $gf_embedding_script \
#         --data_dir "$immune_cell_type_dir/$cell_type" \
#         --out_dir "$immune_cell_type_dir/$cell_type/embeddings/geneformer/aracne_${ARACNE_N_HVG}" \
#         --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
#         --sample_n_cells 1000
# done
