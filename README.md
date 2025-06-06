# scGraphLLM
Graph-based single cell LLM for cell state prediction under perturbations

# TODO

These are some of the tasks/decisions we'll need to make

## Data

We'll need write a data-processing pipeline that will convert the raw scRNA-seq data into generate a list of expressed genes in each cell, with the expression converted into ranks, or bins. In addition to this we'll need to generate the gene-gene graph using Aracne. We'll need to do this for both the internal data from the califano lab, and external data from published papers. For the external data, the Geneformer paper provides [links](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06139-9/MediaObjects/41586_2023_6139_MOESM4_ESM.xlsx) to the papers it gets data from. We can grab some of the data that seems relevant from there on GEO. 

Once we get data preprocessed, we'll need to get the dataloading code written. 

example code to generate cache for modeldata 

```
python scGraphLLM/data.py --aracane-outdir-file pilot_data_dirs.txt --gene-to-node-file /burg/pmg/collab/scGraphLLM/data/cellxgene_gene2index.csv --cache-dir /burg/pmg/collab/scGraphLLM/data/pilotdata_train_cache/ --num-proc 5
mkdir -p /burg/pmg/collab/scGraphLLM/data/pilotdata_valHOG_cache/ 
mkdir -p /burg/pmg/collab/scGraphLLM/data/pilotdata_valSG_cache/ 
mv /burg/pmg/collab/scGraphLLM/data/pilotdata_train_cache/mast_cell* /burg/pmg/collab/scGraphLLM/data/pilotdata_valHOG_cache/
find /burg/pmg/collab/scGraphLLM/data/pilotdata_train_cache/ -type f | shuf -n 12800 | xargs -I {} mv {} /burg/pmg/collab/scGraphLLM/data/pilotdata_valSG_cache/
```

## Install deps

probably a good idea to use a common conda env for all the packages
```
module load cuda12.1
mamba create -y --prefix /pmglocal/$USER/mambaforge/envs/scllm pytorch torchvision torchaudio pytorch-cuda=12.1 pyg lightning pyarrow numpy==1.26.0 -c pyg -c pytorch -c nvidia -c conda-forge
mamba activate /pmglocal/$USER/mambaforge/envs/scllm
pip install ninja scanpy plotnine pandas scikit-learn ipykernel wandb polars fast_matrix_market jupyter loralib
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install flash-attn --no-build-isolation
```


## Inference

To generate single-cell embeddings with our trained model:

1. **Preprocess your data** – We recommend CPM normalization followed by `log1p` transformation.
2. **Generate a gene regulatory network** – Any method can be used, though we recommend [ARACNe](https://califano.c2b2.columbia.edu/aracne/) for consistent results.
3. **Follow our API** – Use our `GeneVocab`, `RegulatoryNetwork`, and `GraphTokenizer` interfaces to convert your data into model-ready format.

Below is a sample Python script using our inference API:

```python
import pandas as pd
from scGraphLLM import GDTransformer, RegulatoryNetwork, GeneVocab, GraphTokenizer, InferenceDataset
from scGraphLLM.config import graph_kernel_attn_3L_4096
from scGraphLLM.inference import get_cell_embeddings

# Load single-cell expression data
data = pd.read_csv("your_data.h5ad")

# Load vocab
vocab = GeneVocab.from_csv("vocab.csv")

# Load your regulatory network
network = RegulatoryNetwork.from_csv("your_network_file.tsv", sep="\t")

# Load trained model checkpoint
model = GDTransformer.load_from_checkpoint("path_to_model.ckpt", config=graph_kernel_attn_3L_4096)

# Create tokenizer from vocab and regulatory network
tokenizer = GraphTokenizer(vocab=vocab, network=network)

# Create dataset for inference
dataset = InferenceDataset(expression=data, tokenizer=tokenizer)

# Run inference and get pooled cell embeddings
embeddings_df = get_cell_embeddings(dataset, model)
