#!/bin/bash

#SBATCH --job-name=scgpt_embedding
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load mamba

immune_cell_types=(
  # "cd14_monocytes"
  # "cd16_monocytes"
  # "cd20_b_cells"
  # "cd4_t_cells"
  # "cd8_t_cells"
  # "erythrocytes"
  # "monocyte-derived_dendritic_cells"
  "nk_cells"
  "nkt_cells"
)

immune_cell_type_dir="/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type"

# ARACNE_N_HVG=1024
ARACNE_N_HVG=4096

# # scGraphLLM with entire human immune dataset
# mamba deactivate
# mamba activate scllm

# scglm_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/scglm_embedding.py"

# echo "Querying scGraphLLM embeddings from processed immune cells..."

# python $scglm_embedding_script \
#   --cells_path /hpc/mydata/rowan.cassius/data/scGPT/human_immune/processed_human_immune.h5ad \
#   --out_dir /hpc/mydata/rowan.cassius/data/scGPT/human_immune/embeddings/scglm/aracne_${ARACNE_N_HVG}_cls_3L_12000_steps_MLM_002_infer_network \
#   --networks human_immune_metacells_networks \
#   --infer_network \
#   --model_path "/hpc/mydata/leo.dupire/GLM/model_out/PRETRAIN [CLS, 3Layer, 3Diff, lr:5e-5, Q-mixing]:2025-05-08@05:12:51/checkpoints/epoch=0-step=45000.ckpt" \
#   --gene_index_path /hpc/projects/group.califano/GLM/data/cellxgene_gene2index_with_cls.csv \
#   --retain_obs_vars final_annotation batch set sample_id obs_id \
#   --max_seq_length 2048 \
#   --cache


# 001 model path
# --model_path "/hpc/mydata/leo.dupire/GLM/model_out/PRETRAIN [CLS, 3Layer, rank_mask:15%, lr:0.00005, AdamW]:2025-05-05@01:26:24/checkpoints/epoch=0-step=12000.ckpt" \

#==============#
#=*= scGPT #=*=#
#==============#

# mamba deactivate
# mamba activate scgpt

# scgpt_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/scgpt_embedding.py"

# for cell_type in "${immune_cell_types[@]}"; do
#     echo "Querying scGPT embeddings for cell type: $cell_type..."
    
#     python $scgpt_embedding_script \
#         --data_dir "$immune_cell_type_dir/$cell_type" \
#         --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scgpt/aracne_${ARACNE_N_HVG}_seq_len_2048" \
#         --model_dir /hpc/mydata/rowan.cassius/scGPT/scGPT_human \
#         --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
#         --scgpt_rootdir /hpc/mydata/rowan.cassius/scGPT \
#         --retain_obs_vars final_annotation batch sample_id obs_id \
#         --sample_n_cells 1000 \
#         --cache
# done

#=====================#
#=*= scFoundation #=*=#
#=====================#

mamba deactivate
mamba activate scllm

scf_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/scfoundation_embedding.py"

for cell_type in "${immune_cell_types[@]}"; do
    echo "Querying scFoundation embeddings for cell type: $cell_type..."
    
    python $scf_embedding_script \
        --data_dir "$immune_cell_type_dir/$cell_type" \
        --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scfoundation/aracne_${ARACNE_N_HVG}_seq_len_2048" \
        --model_path /hpc/mydata/rowan.cassius/scFoundation/model/models/models.ckpt \
        --gene_index_path /hpc/mydata/rowan.cassius/scFoundation/model/OS_scRNA_gene_index.19264.tsv \
        --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
        --scf_rootdir /hpc/mydata/rowan.cassius/scFoundation \
        --retain_obs_vars final_annotation batch sample_id obs_id \
        --sample_n_cells 1000 \
        --max_seq_length 2048 \
        --cache
done

# ===================#
# =*= scGraphLLM #=*=#
# ===================#

# mamba deactivate
# mamba activate scllm

# scglm_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/scglm_embedding.py"

# for cell_type in "${immune_cell_types[@]}"; do
#     echo "Querying scGraphLLM embeddings for cell type: $cell_type..."

#     # Without Edge Masks
#     python $scglm_embedding_script \
#         --data_dir "$immune_cell_type_dir/$cell_type" \
#         --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scglm/aracne_${ARACNE_N_HVG}_cls_3L_12000_steps_MLM_001_correct_expression" \
#         --model_path "/hpc/mydata/leo.dupire/GLM/model_out/PRETRAIN [CLS, 3Layer, rank_mask:15%, lr:0.00005, AdamW]:2025-05-05@01:26:24/checkpoints/epoch=0-step=12000.ckpt" \
#         --gene_index_path /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv \
#         --network_path "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG/consolidated-net_defaultid.tsv" \
#         --retain_obs_vars final_annotation batch sample_id obs_id \
#         --sample_n_cells 1000 \
#         --cache

    # # With Edge Masks
    # python $scglm_embedding_script \
    #     --data_dir "$immune_cell_type_dir/$cell_type" \
    #     --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scglm/aracne_${ARACNE_N_HVG}_cls_3L_12000_steps_MLM_001_edge_mask_0.15" \
    #     --model_path "/hpc/mydata/leo.dupire/GLM/model_out/PRETRAIN [CLS, 3Layer, rank_mask:15%, lr:0.00005, AdamW]:2025-05-05@01:26:24/checkpoints/epoch=0-step=12000.ckpt" \
    #     --gene_index_path /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv \
    #     --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
    #     --retain_obs_vars final_annotation batch sample_id obs_id \
    #     --sample_n_cells 1000 \
    #     --cache \
    #     --use_masked_edges \
    #     --mask_fraction 0.15 \
    #     --use_masked_edges

    # # Without Integrated Network
    # python $scglm_embedding_script \
    #     --data_dir "$immune_cell_type_dir/$cell_type" \
    #     --out_dir "$immune_cell_type_dir/$cell_type/embeddings/scglm/aracne_${ARACNE_N_HVG}_cls_3L_12000_steps_MLM_001_integrated_network" \
    #     --model_path "/hpc/mydata/leo.dupire/GLM/model_out/PRETRAIN [CLS, 3Layer, rank_mask:15%, lr:0.00005, AdamW]:2025-05-05@01:26:24/checkpoints/epoch=0-step=12000.ckpt" \
    #     --gene_index_path /hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv \
    #     --network_path /hpc/mydata/rowan.cassius/data/scGPT/human_immune/network/integrated_network.tsv \
    #     --retain_obs_vars final_annotation batch sample_id obs_id \
    #     --sample_n_cells 1000 \
    #     --cache 
# done


# ===================#
# =*= Geneformer #=*=#
# ===================#

# mamba deactivate
# mamba activate llm

# gf_embedding_script="/hpc/mydata/rowan.cassius/scGraphLLM/embeddings/geneformer_embedding.py"

# for cell_type in "${immune_cell_types[@]}"; do
#     echo "Querying scGraphLLM embeddings for cell type: $cell_type..."
    
#     python $gf_embedding_script \
#         --data_dir "$immune_cell_type_dir/$cell_type" \
#         --out_dir "$immune_cell_type_dir/$cell_type/embeddings/geneformer/aracne_${ARACNE_N_HVG}_seq_len_2048" \
#         --aracne_dir "$immune_cell_type_dir/$cell_type/aracne_$ARACNE_N_HVG" \
#         --retain_obs_vars final_annotation batch sample_id obs_id \
#         --sample_n_cells 1000 \
#         --cache
# done