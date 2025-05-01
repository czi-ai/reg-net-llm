## these correspond to indices in the gene and rank embedding layers 
PAD_GENE_IDX=0
MASK_GENE_IDX=1

ZERO_IDX=0 ## ONLY APPLIES TO EXPRESSION TOKENS: the lowest expression, aka gene with expression 0
MIN_GENES_PER_GRAPH=500 # Shouldn't this be 1024 ?
NUM_GENES=19246 # this is the nubmer of genes + special tokens
NUM_BINS=100
NUM_RANKS=NUM_BINS + 2 ## this is the number of ranks + special tokens
PAD_RANK_IDX=(NUM_BINS-1) + 1 ## the padding token
MASK_RANK_IDX=(NUM_BINS-1) + 2 ## the mask token