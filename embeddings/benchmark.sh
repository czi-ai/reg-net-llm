#!/bin/bash

#SBATCH --time=02:00:00
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

# (This is only meaningful for scGLM)
python $benchmark_script \
  --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scglm \
  --model scglm \
  --suffix aracne_4096_mask_0.45 \
  --task mgm

#================================#
#=*= MGM using GATConv Layer  #=*#
#================================#

# mask_ratio=0.3

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scglm \
#   --model scglm \
#   --suffix aracne_4096_gat \
#   --task mgm \
#   --use_gat

python $benchmark_script \
  --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/geneformer \
  --model gf \
  --suffix aracne_4096_mask_0.45_gat \
  --task mgm \
  --use_gat \
  --generate_edge_masks \
  --mask_ratio 0.45

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scgpt \
#   --model scgpt \
#   --suffix aracne_4096_gat \
#   --task mgm \
#   --use_gat \
#   --generate_edge_masks

# python $benchmark_script \
#   --out_dir /hpc/mydata/rowan.cassius/tasks/mgm/scfoundation \
#   --model scf \
#   --suffix aracne_4096_gat \
#   --task mgm \
#   --use_gat \
#   --generate_edge_masks 