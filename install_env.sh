module load cuda
module load mamba

mamba env create --file env.yml
conda activate scllm
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html