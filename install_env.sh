#!/bin/bash
rm -rf $MAMBA_ROOT_PREFIX/scllm
mamba env create --prefix $MAMBA_ROOT_PREFIX/scllm --file env.yml
## must run this command below to install pyg-lib 
echo "run  pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html "