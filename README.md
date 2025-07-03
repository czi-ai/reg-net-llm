# GREmLN: Graph structure aware foundation model for scRNA data
The code base for the graph-based single cell foundation model developed at CZB NY. The goal is to learn meaningful foundational gene embeddings to faciliate downstream tasks such as perturbation prediction through incoperating gene regulatory network topology into the transformer attention mechnism. 

Ownership: Califano Lab \
Main developers: Mingxuan Zhang, Vinay Swamy, Rowan Cassius, Léo Dupire

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).

## Install deps
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
from scGraphLLM import RegulatoryNetwork, GeneVocab, GraphTokenizer, InferenceDataset
from scGraphLLM.models import GDTransformer
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

# Create tokenizer and dataset for inference
dataset = InferenceDataset(
  expression=data,
  tokenizer=GraphTokenizer(vocab=vocab, network=network) 
)

# Run inference and get pooled cell embeddings
embeddings_df = get_cell_embeddings(dataset, model)

```

Custom Tokenization Example

```python
# prune network
network = RegulatoryNetwork.from_csv("your_network_file.tsv", sep="\t")\
  .prune(limit_regulon=100)\
  .make_undirected(drop_unpaired=False)

tokenizer = GraphTokenizer(
  vocab=vocab,
  network=network,
  only_expressed_genes=False, 
  max_seq_length=4096,
  with_edge_weights=True
)

```

