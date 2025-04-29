import os
import torch
import sys

dataset_name = sys.argv[1]
cache_dir = sys.argv[2]
process_id = int(sys.argv[3])-1

DS = cache_dir.split("/")[-1]
os.makedirs(f"/hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/corrupted_files/{DS}_corrupt_{dataset_name}", exist_ok=True)
a = open(f"/hpc/projects/group.califano/GLM/scGraphLLM/scripts/ID_corrupted/corrupted_files/{DS}_corrupt_{dataset_name}/{process_id}.txt", "w")
path = f"{cache_dir}/{dataset_name}"

paths = os.listdir(path)
window = round(len(paths)/100) + 1
start = window * process_id
end = window * (process_id + 1)

if end > len(paths):
    end = len(paths)

for f in paths[start:end]:
    pt = os.path.join(path, f)
    try:
        tmp = torch.load(pt)
        del tmp
    except:
        print(pt)
        a.write(pt)
        a.write("\n")