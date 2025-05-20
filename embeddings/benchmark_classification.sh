#!/bin/bash

#SBATCH --job-name=benchmark
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4


module load mamba
mamba activate scllm

benchmark_script="/hpc/mydata/rowan.cassius/scGraphLLM/scGraphLLM/benchmark.py"


# ===================#
# =*= Geneformer #=*=#
# ===================#

# python $benchmark_script \
#     --dataset human_immune_geneformer_seq_len_2048 \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/geneformer \
#     --suffix human_immune_geneformer_seq_len_2048 \
#     --use_weighted_ce \
#     --task cls \
#     --target final_annotation \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5

# python $benchmark_script 
#     --dataset gf \
#     --split_config mye \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/geneformer \
#     --suffix mye_split_gat \
#     --task cls \
#     --use_gat \
#     --target cell_type \
#     --lr 1e-2 \
#     --num_epochs 100 \
#     --patience 5


# Myeloid Dataset
# python $benchmark_script \
#     --dataset gf_seq_len_2048 \
#     --split_config mye \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/geneformer \
#     --suffix mye_gf_seq_len_2048_linear_good_graph \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5


# ===================#
# =*= scGraphLLM #=*=#
# ===================#

# # Human Immune Dataset 
# python $benchmark_script \
#     --dataset human_immune_scglm_cls_3L_12000_steps_MLM_002_infer_network \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix human_immune_scglm_cls_3L_12000_steps_MLM_002_infer_network_linear \
#     --use_weighted_ce \
#     --task cls \
#     --target final_annotation \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5

# Myeloid Dataset
python $benchmark_script \
    --dataset mye_scglm_MLM_001_regulon_pruned \
    --split_config mye \
    --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
    --suffix mye_scglm_MLM_001_regulon_pruned \
    --use_weighted_ce \
    --task cls \
    --target cell_type \
    --lr 1e-2 \
    --batch_size 256 \
    --num_epochs 200 \
    --patience 5


# python $benchmark_script \
#     --dataset scglm \
#     --split_config mye \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix mye_split_gat \
#     --task cls \
#     --use_gat \
#     --target cell_type \
#     --lr 1e-2 \
#     --num_epochs 100 \
#     --patience 5

# Pancreas Dataset
# python $benchmark_script \
#     --dataset pancreas_scglm_cls_3L_12000_steps_MLM_001_pruned_graph_002 \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix pancreas_scglm_cls_3L_12000_steps_MLM_001_pruned_graph_002_linear \
#     --use_weighted_ce \
#     --task cls \
#     --cls_layers 1 \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5

# Adamson Dataset
# python $benchmark_script \
#     --dataset adamson_scglm_cls_3L_12000_steps_MLM_001_pruned_graph_100 \
#     --split_config adamson \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix adamson_scglm_cls_3L_12000_steps_MLM_001_pruned_graph_100 \
#     --use_weighted_ce \
#     --task cls \
#     --cls_layers 3 \
#     --target condition \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 400 \
#     --patience 20


# ==============#
# =*= scGPT #=*=#
# ==============#

# # Human Immune
# python $benchmark_script \
#     --dataset human_immune_scgpt_seq_len_2048 \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scgpt \
#     --suffix human_immune_scgpt_seq_len_2048_linear \
#     --use_weighted_ce \
#     --task cls \
#     --target final_annotation \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5


# # human immune scf
# python $benchmark_script \
#     --dataset scf_human_immune \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scfoundation \
#     --suffix scf_human_immune_linear \
#     --use_weighted_ce \
#     --task cls \
#     --target final_annotation \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5



# # Myeloid Dataset
# python $benchmark_script \
#     --dataset scgpt_seq_len_2048 \
#     --split_config mye \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scgpt \
#     --suffix mye_scgpt_seq_len_2048_linear_good_graph \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5


# # Pancreas Dataset
# python $benchmark_script \
#     --dataset pancreas_scgpt_seq_len_2048 \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scgpt \
#     --suffix pancreas_scgpt_seq_len_2048_linear \
#     --use_weighted_ce \
#     --task cls \
#     --cls_layers 1 \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5




######## BRAIN ############

# Brain Dataset
# python $benchmark_script \
#     --dataset scglm_brain \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix scglm_brain \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5

# python $benchmark_script \
#     --dataset scgpt_brain \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scgpt \
#     --suffix scgpt_brain \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5

# python $benchmark_script \
#     --dataset gf_brain \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/geneformer \
#     --suffix gf_brain \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5





####### HOG ########

# python $benchmark_script \
#     --dataset gf_hog \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/geneformer \
#     --suffix gf_hog \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5


# python $benchmark_script \
#     --dataset scglm_hog \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix scglm_hog \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5



# python $benchmark_script \
#     --dataset scgpt_hog \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scgpt \
#     --suffix scgpt_hog \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5


# python $benchmark_script \
#     --dataset scf_hog \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scfoundation \
#     --model_path /hpc/mydata/rowan.cassius/tasks/cls/scfoundation/scf_hog_2025-05-15_18-09/model/epoch=8-step=234-val_loss=0.2959.ckpt \
#     --prediction \
#     --suffix scf_hog \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 200 \
#     --patience 5




