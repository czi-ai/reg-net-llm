#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4

module load mamba
mamba activate scllm

benchmark_script="/hpc/mydata/rowan.cassius/scGraphLLM/scGraphLLM/benchmark.py"

echo "Running Benchmark Script..."


# for more time --time=1-00:00:00

#===============================================#
#=*= Link Prediction using Frozen Embeddings #=*#
#===============================================#

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/scglm \
#   --model scglm \
#   --suffix aracne_4096 \
#   --task link

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/geneformer \
#   --model gf \
#   --suffix aracne_4096 \
#   --task link

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/scfoundation \
#   --model scf \
#   --suffix aracne_4096 \
#   --task link

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/scgpt \
#   --model scgpt \
#   --suffix aracne_4096 \
#   --task link

#============================================#
#=*= Link Prediction using GATConv Layer  #=*#
#============================================#
  
# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/scglm \
#   --model scglm \
#   --suffix aracne_4096_gat \
#   --task link \
#   --use_gat

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/scfoundation \
#   --model scf \
#   --suffix aracne_4096_gat \
#   --task link \
#   --use_gat

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/scgpt \
#   --model scgpt \
#   --suffix aracne_4096_gat \
#   --task link \
#   --use_gat

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/link_prediction/geneformer \
#   --model gf \
#   --suffix aracne_4096_gat \
#   --task link \
#   --use_gat

#====================================#
#=*= MGM using Frozen Embeddings  #=*#
#====================================#

# # (This is only meaningful for scGLM)
# python $benchmark_script \
#   --dataset human_immune_scglm_cls_3L_12000_steps_MLM_001_edge_mask_0.15 \
#   --split_config human_immune \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scglm \
#   --suffix aracne_4096_cls_3L_12000_steps_MLM_001_edge_mask_0.15 \
#   --task mgm \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5

# # myeloid
# python $benchmark_script \
#   --dataset mye_scglm_cls_3L_12000_steps_MLM_001_seq_len_2048_mask_0.15 \
#   --split_config mye \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scglm \
#   --suffix mye_scglm_cls_3L_12000_steps_MLM_001_seq_len_2048_mask_0.15 \
#   --task mgm \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5


#=================================#
#=*= MGM using GATConv Layer  #=*=#
#=================================#

mask_ratio=0.15

# # human immune geneformer
# python $benchmark_script \
#   --dataset human_immune_geneformer_seq_len_2048 \
#   --split_config human_immune \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/geneformer \
#   --suffix human_immune_geneformer_seq_len_2048_mask_$mask_ratio \
#   --task mgm \
#   --mask_ratio $mask_ratio \
#   --use_gat \
#   --generate_edge_masks \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5

# myeloid geneformer
# python $benchmark_script \
#   --dataset gf_seq_len_2048 \
#   --split_config mye \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/geneformer \
#   --suffix mye_gf_seq_len_2048_mask_$mask_ratio \
#   --task mgm \
#   --mask_ratio $mask_ratio \
#   --use_gat \
#   --generate_edge_masks \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5


# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scglm \
#   --model scglm \
#   --suffix aracne_4096_gat \
  # --task mgm \
  # --use_gat

# # human immune
# python $benchmark_script \
#   --dataset human_immune_scgpt_seq_len_2048 \
#   --split_config human_immune \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scgpt \
#   --suffix human_immune_scgpt_seq_len_2048_mask_0.15 \
#   --task mgm \
#   --mask_ratio $mask_ratio \
#   --use_gat \
#   --generate_edge_masks \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5

# # myeloid
# python $benchmark_script \
#   --dataset scgpt_seq_len_2048 \
#   --split_config mye \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scgpt \
#   --suffix mye_scgpt_seq_len_2048_mask_$mask_ratio \
#   --task mgm \
#   --mask_ratio $mask_ratio \
#   --use_gat \
#   --generate_edge_masks \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5


# human immune scf
# python $benchmark_script \
#   --dataset scf_human_immune \
#   --split_config human_immune \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scfoundation \
#   --suffix scf_human_immune_mask_0.15 \
#   --task mgm \
#   --mask_ratio $mask_ratio \
#   --use_gat \
#   --generate_edge_masks \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5


python $benchmark_script \
  --dataset scf_mye \
  --split_config mye \
  --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scfoundation \
  --suffix scf_mye_mask_$mask_ratio \
  --task mgm \
  --mask_ratio $mask_ratio \
  --use_gat \
  --generate_edge_masks \
  --lr 1e-3 \
  --num_epochs 200 \
  --patience 5 


# python $benchmark_script \
#   --dataset scgpt_seq_len_2048 \
#   --split_config mye \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scgpt \
#   --suffix mye_scgpt_seq_len_2048_mask_$mask_ratio \
#   --task mgm \
#   --mask_ratio $mask_ratio \
#   --use_gat \
#   --generate_edge_masks \
#   --lr 1e-3 \
#   --num_epochs 200 \
#   --patience 5


# human immmune
# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scfoundation \
#   --model scf \
#   --suffix aracne_4096_gat \
#   --task mgm \
#   --use_gat \
#   --generate_edge_masks 