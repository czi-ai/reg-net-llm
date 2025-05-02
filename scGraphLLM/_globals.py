## these correspond to indices in the gene and rank embedding layers 
PAD_GENE_IDX=0
MASK_GENE_IDX=1
CLS_GENE_IDX=19246

ZERO_IDX=0 ## ONLY APPLIES TO EXPRESSION TOKENS: the lowest expression, aka gene with expression 0
MIN_GENES_PER_GRAPH=500 
#NUM_GENES=19246 # this is the nubmer of genes + special tokens
NUM_GENES=19247 ## this is the nubmer of genes + special tokens
NUM_BINS=100
NUM_RANKS=NUM_BINS + 3 ## this is the number of ranks + special tokens
PAD_RANK_IDX=(NUM_BINS-1) + 1 ## the padding token
MASK_RANK_IDX=(NUM_BINS-1) + 2 ## the mask token
CLS_TOKEN = (NUM_BINS-1) + 3 # CLS token
BETA = 0.1 # diffusion rate

