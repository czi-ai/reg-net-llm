# Download the latest Miniforge installer with curl
#curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh

# Install miniforge3 and initiate it
#bash Miniforge3.sh -b -p /work/miniforge3
/mnt/pvc/miniforge3/bin/python /mnt/pvc/miniforge3/bin/conda init bash
source ~/.bashrc

# Delete installer
#rm Miniforge3.sh

# Create Conda env
#mamba create -n gremln pytorch torchvision torchaudio pyg lightning pyarrow numpy==1.26.0 ninja scanpy plotnine pandas scikit-learn ipykernel wandb polars fast_matrix_market jupyter loralib pyg-lib flash-attn -c conda-forge -c pytorch -c nvidia -c pyg

# Activate gremln
#conda activate gremln
