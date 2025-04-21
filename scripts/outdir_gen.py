# This file generates the outdirs file and allots dirs SG, and HOG. The output file is used to cache the data for training in data.py.

import os
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--cxg-path", type=str, required=True) # /hpc/projects/group.califano/GLM/data/cellxgene/data/complete_data_ranked_z_score
parser.add_argument("--outfile-path", type=str, required=True) # /hpc/projects/group.califano/GLM/data/aracne_4096_outdir.csv
parser.add_argument("--slurm-out-path", type=str, required=True) # /hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/1024_z_scored
parser.add_argument("--aracne-top-n-hvg", type=int, required=True) # 1024
args = parser.parse_args()

# python outdir_gen.py --cxg-dir /hpc/projects/group.califano/GLM/data/cellxgene/data/complete_data_ranked_z_score --slurm-out-path /hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/2048_z_scored --aracne-top-n-hvg 2048

CXG_PATH = args.cxg_path
OUTFILE_PATH = args.outfile_path
aracne_top_n_hvg = args.aracne_top_n_hvg
cell_types = [dir[:-1] for dir in open(f"{args.slurm_out_path}/successful_cell_types.txt", "r")]

# Enter the proportions for each dataset
SG_prop = 0.85
HOG_prop = 0.15

# Check file size and delegate the dirs according to their size rather than equal weighting of each
SG_num = int(len(cell_types)//(1/SG_prop))
HOG_num = int(len(cell_types) - SG_num)

print("SG_num:", SG_num)
print("HOG_num:", HOG_num)

# Shuffle the cell-type list so the assignment below is random
random.shuffle(cell_types)

# Iterate through cell-types and designate if seen (SG) or not seen (HOG)
outdirs = open(OUTFILE_PATH, "w")
for i in range(SG_num): # Assign SG
    outdirs.write(f"{CXG_PATH}/{cell_types[i]}/aracne_{aracne_top_n_hvg},valSG\n")
    
for i in range(SG_num, SG_num+HOG_num): # Assign HOG
    outdirs.write(f"{CXG_PATH}/{cell_types[i]}/aracne_{aracne_top_n_hvg},valHOG\n")
    
# End program
outdirs.close()
print("Done!")
    