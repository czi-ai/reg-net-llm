import os
import torch
import sys

corrupt_dir = sys.argv[1]

for dataset in os.listdir(corrupt_dir):
    for path in os.listdir(f"{corrupt_dir}/{dataset}"):
        file = open(f"{corrupt_dir}/{dataset}/{path}", "r")
        txt = [line for line in file]
        print(txt)
        for pt in txt:
            if os.path.exists(pt):
                os.remove(pt)
                print("Deleted")
        file.close()
        del txt
    
# python /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/delete_corrupt.py /hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/corrupted_files