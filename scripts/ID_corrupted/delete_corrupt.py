import os
import torch
import sys

corrupt_dir = sys.argv[1]

for dataset in os.listdir(corrupt_dir):
    print("-"*10, dataset)
    for path in os.listdir(f"{corrupt_dir}/{dataset}"):
        file = open(f"{corrupt_dir}/{dataset}/{path}", "r")
        txt = [line for line in file]
        for pt in txt:
            pt = pt[:-1]
            print(pt, end="") # Exclude the last element (newline)
            if "__conda_exe: command not found" in pt: # Artifact
                continue
            
            if os.path.exists(pt):
                os.remove(pt)
                print("\t----\tDeleted")
            else:
                print()
        file.close()
        del txt
    print()
    
# python /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/delete_corrupt.py /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/corrupted_files