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

# Human Immune Dataset 
python $benchmark_script \
    --dataset human_immune_scglm_cls_3L_12000_steps_MLM_001_repro \
    --split_config random \
    --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
    --suffix human_immune_scglm_cls_3L_12000_steps_MLM_001_repro_repro \
    --use_weighted_ce \
    --task cls \
    --target final_annotation \
    --lr 1e-2 \
    --batch_size 256 \
    --num_epochs 100 \
    --patience 5

# Myeloid Dataset
# python $benchmark_script \
#     --dataset scglm_graph_metacells \
#     --split_config mye \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix mye_scglm_graph_metacells \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5


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
#     --dataset pancreas_scglm_cls_3L_12000_steps_MLM_001 \
#     --split_config train_test_set \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
#     --suffix pancreas_scglm_cls_3L_12000_steps_MLM_001 \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5


# ==============#
# =*= scGPT #=*=#
# ==============#

# python $benchmark_script \
#     --dataset human_immune_scgpt_seq_len_2048 \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scgpt \
#     --suffix human_immune_scgpt_seq_len_2048 \
#     --use_weighted_ce \
#     --task cls \
#     --target final_annotation \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5


# Myeloid Dataset
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
