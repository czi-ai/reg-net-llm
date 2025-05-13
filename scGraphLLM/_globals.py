## these correspond to indices in the gene and rank embedding layers 
PAD_GENE_IDX=0
MASK_GENE_IDX=1
CLS_GENE_IDX=19246

ZERO_IDX=0 ## ONLY APPLIES TO BINNED (NOT RAW) EXPRESSION TOKENS: the lowest expression, aka gene with expression 0
MIN_GENES_PER_GRAPH=500 
#NUM_GENES=19246 # This is the nubmer of genes + special tokens
NUM_GENES=19247 # This is the nubmer of genes + special tokens
NUM_BINS=100 # Number of bins that the raw expression is quantized into
NUM_EXPRESSION_BINS=NUM_BINS + 3 # This is the number of ranks + special tokens
PAD_EXPRESSION_IDX=(NUM_BINS-1) + 1 # The padding token
MASK_EXPRESSION_IDX=(NUM_BINS-1) + 2 # The mask token
CLS_TOKEN = (NUM_BINS-1) + 3 # CLS token
BETA = 0.1 # Diffusion rate

