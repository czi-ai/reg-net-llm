# This file generates the outdirs file and allots dirs to train, ValSG, and ValHG. The output file is used to cache the data for training in data.py.

import os

# Enter the proportions for each dataset: default is train: 80%, valSG: 10%, valHG: 10%
TRAIN_prop = 0.8
VAL_SG_prop = 0.1
VAL_HG_prop = 0.1

ARACNE_TOP_N_HVG = 1024 # CHANGE THE FOLLOWING TO YOUR SPECIFICATIONS
run = "16726641" # THIS IS THE JOB ID from the `start_slurm_preprocess.sh` run you would like to use


success_file = open(f"/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/slurm_out_{run}/check_out/success.txt", "r")
success_file = [dir[:-1] for dir in success_file]

# Check file size and delegate the dirs according to their size rather than equal weighting of each
TRAIN_num = int(len(success_file)//(1/TRAIN_prop))
SG_num = int(len(success_file)//(1/VAL_SG_prop))
HG_num = int(len(success_file) - TRAIN_num - SG_num)

print("TRAIN_num:", TRAIN_num)
print("VAL_SG_num:", SG_num)
print("VAL_HG_num:", HG_num)

outdirs = open(f"/hpc/projects/group.califano/GLM/data/aracne_{ARACNE_TOP_N_HVG}_outdir.csv", "w")
aracne_path = "/hpc/projects/group.califano/GLM/data/cellxgene/data/complete_data"

for i in range(TRAIN_num):
    outdirs.write(f"{aracne_path}/{success_file[i]}/aracne_{ARACNE_TOP_N_HVG},train\n")

for i in range(TRAIN_num, TRAIN_num+SG_num):
    outdirs.write(f"{aracne_path}/{success_file[i]}/aracne_{ARACNE_TOP_N_HVG},valSG\n")
    
for i in range(TRAIN_num+SG_num, TRAIN_num+SG_num+HG_num):
    outdirs.write(f"{aracne_path}/{success_file[i]}/aracne_{ARACNE_TOP_N_HVG},valHG\n")
    
outdirs.close()
print("Done!")
    