#!/bin/bash
#SBATCH --account pmg
## make sure you run mamba init at least once before running this 
set -e
rm -rf ~/.local
if [ -d "/pmglocal/$USER/mambaforge/envs" ]; then
    echo "mambaforge envs directory already exits"
else
    mkdir -p /pmglocal/$USER/mambaforge/envs
fi
module load cuda12.1
module load mamba
# hard rest of virual environment
rm -rf /pmglocal/$USER/mambaforge/envs/scllm
mamba create -y --prefix /pmglocal/$USER/mambaforge/envs/scllm pytorch torchvision torchaudio pytorch-cuda=12.1 pyg lightning pyarrow -c pyg -c pytorch -c nvidia -c conda-forge
source activate /pmglocal/$USER/mambaforge/envs/scllm
which pip 
pip install ninja scanpy plotnine pandas scikit-learn ipykernel wandb polars fast_matrix_market jupyter loralib
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install flash-attn --no-build-isolation