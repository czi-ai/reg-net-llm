# scGraphLLM
Graph-based single cell LLM for cell state prediction under perturbations

# TODO

These are some of the tasks/decisions we'll need to make

## Data

We'll need write a data-processing pipeline that will convert the raw scRNA-seq data into generate a list of expressed genes in each cell, with the expression converted into ranks, or bins. In addition to this we'll need to generate the gene-gene graph using Aracne. We'll need to do this for both the internal data from the califano lab, and external data from published papers. For the external data, the Geneformer paper provides [links](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06139-9/MediaObjects/41586_2023_6139_MOESM4_ESM.xlsx) to the papers it gets data from. We can grab some of the data that seems relevant from there on GEO. 

Once we get data preprocessed, we'll need to get the dataloading code written. 


## Model

One of the challenges will be figuring out the exact training procedure. One option would be for each *batch* of cells, we first update the whole graph with the GNN, extract the gene embeddings, and then run the transformer block using these embeddings. This is contingent on how quick it is to update the graph.

If updating the graph is too slow, we can try to update the graph every *epoch* instead, where each epoch is split in two 2 phases, a graph learning phase, where we feed batches of the graph to the GNN, and then a second phase where we feed batches of the cells to the transformer block.

## Install deps

probably a good idea to use a common conda env for all the packages
```
mamba create -n scllm pytorch torchvision torchaudio pytorch-cuda=12.1 pyg lightning -c pyg -c pytorch -c nvidia -c conda-forge
pip install ninja scanpy plotnine pandas scikit-learn ipykernel wandb 

```