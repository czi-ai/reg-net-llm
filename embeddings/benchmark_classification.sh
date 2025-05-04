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
#     --dataset gf_seq_len_512 \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/geneformer \
#     --suffix mye_512_random_split_2_layers_mean_pool \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
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


# ===================#
# =*= scGraphLLM #=*=#
# ===================#

python $benchmark_script \
    --dataset scglm_seq_len_512 \
    --split_config random \
    --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scglm \
    --suffix mye_512_random_split_2_layers_mean_pool \
    --use_weighted_ce \
    --task cls \
    --target cell_type \
    --lr 1e-2 \
    --batch_size 256 \
    --num_epochs 100 \
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


# ==============#
# =*= scGPT #=*=#
# ==============#

# python $benchmark_script \
#     --dataset scgpt_seq_len_512 \
#     --split_config random \
#     --out_dir /hpc/mydata/rowan.cassius/tasks/cls/scgpt \
#     --suffix mye_512_random_split_2_layers_mean_pool \
#     --use_weighted_ce \
#     --task cls \
#     --target cell_type \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_epochs 100 \
#     --patience 5