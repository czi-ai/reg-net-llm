import os
import torch
import sys

if len(sys.argv) > 1:
    cache_dir = sys.argv[1]
    
DS = cache_dir.split("/")[-1]
a = open(f"./corrupted_files/{DS}_corrupt_SG.txt", "w")
path = f"{cache_dir}/valSG"
for f in os.listdir(path):
    pt = os.path.join(path, f)
    try:
        tmp = torch.load(pt)
        del tmp
    except:
        print(pt)
        a.write(pt)
        a.write("\n")