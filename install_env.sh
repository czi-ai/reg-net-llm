#!/bin/bash
## make sure you run mamba init at least once before running this 
set -e

mkdir -p /pmglocal/$USER/mambaforge/envs/scllm
module load cuda12.1
source activate /pmglocal/$USER/mambaforge/envs/scllm pytorch torchvision torchaudio pytorch-cuda=12.1 pyg lightning pyarrow -c pyg -c pytorch -c nvidia -c conda-forge
mamba activate /pmglocal/$USER/mambaforge/envs/scllm
pip install ninja scanpy plotnine pandas scikit-learn ipykernel wandb polars fast_matrix_market jupyter loralib
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install flash-attn --no-build-isolation