## these correspond to indices in the gene and rank embedding layers 
PAD_GENE_IDX=0
MASK_GENE_IDX=1
CLS_GENE_IDX=19246

CLS_GENE="<CLS>"
MASK_GENE="<MASK>"
PAD_GENE="<PAD>"

ZERO_IDX=0 ## ONLY APPLIES TO EXPRESSION TOKENS: the lowest expression, aka gene with expression 0
MIN_GENES_PER_GRAPH=500
NUM_GENES=19247 # This is the nubmer of genes + special tokens
NUM_BINS=100 # Number of bins that the raw expression is quantized into
NUM_RANKS=NUM_BINS + 3 # This is the number of ranks + special tokens
PAD_RANK_IDX=(NUM_BINS-1) + 1 # The padding token
MASK_RANK_IDX=(NUM_BINS-1) + 2 # The mask token
CLS_RANK_IDX = (NUM_BINS-1) + 3 # CLS token
BETA = 0.1 # Diffusion rate

# Variables from ARACNe
REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"
LOGP_VALS = "log.p.values"
SCC_VALS = "scc.values"
WT_VALS = "weight.values"

# Tokenization Parameters
MAX_SEQ_LENGTH=4096