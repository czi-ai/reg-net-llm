import os 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_slurm_out_path", type=str, required=True) # /hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out/1024_base
args = parser.parse_args()

def extract_cell_type(file):
    return file[0][:-4].split()[-1]

success = []
PATH = args.dataset_slurm_out_path
for dr in os.listdir(PATH):
    if dr == "successful_cell_types.txt": # Success file already present from previous caching run
            continue
    for out_path in os.listdir(os.path.join(PATH, dr)):
        if out_path == "check_out":
            continue
        
        out_path = os.path.join(PATH, dr, out_path)
        out_file = open(out_path, "r")
        out_list = [line for line in out_file]
        out_str = "".join(out_list)
        out_file.close()
        
        cell_type = extract_cell_type(out_list)
        if ("Successfully ran ARACNe" in out_str) and (cell_type not in success): # Check that the processing was successful and completed for this cell-type
            success.append(cell_type)
           
success_file = open(os.path.join(PATH, "successful_cell_types.txt"), "w")
for ct in success:
    success_file.write(f"{ct}\n")
success_file.close()

print(f"Successfully processed cell-types can be found at: {PATH}/successful_cell_types.txt\n")
