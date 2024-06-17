## these correspond to indices in the gene and rank embedding layers 
PAD_IDX=0 ## the padding token
MASK_IDX=1 ## the mask token
ZERO_IDX=2 ## the lowest rank, aka gene with expression 0
MIN_GENES_PER_GRAPH=500
NUM_GENES=5003## this is the nubmer of genes + special tokens 