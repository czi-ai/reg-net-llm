import os 

def extract_cell_type(file):
    return file[0][:-4].split()[-1]

# CHANGE TO YOUR DIRECTORY !
PATH = "/hpc/projects/group.califano/GLM/scGraphLLM/scripts/slurm_out"
dirs = os.listdir(PATH)

job = None
# job = "array_job_786994"
if job == None:
    job_list = list(set([dir[:18] for dir in dirs]))
    job_list.sort()
    print("\nOut-files for the following jobs were found:\n")
    for i in range(len(job_list)):
        print(f"\t{i}. {job_list[i]}")
    
    opt = None
    while type(opt) != int:
        try:
            opt = int(input("\nWhich job would you like to check (0/1/2/...)?   "))
        except:
            print("Please enter a valid integer.")

    job = job_list[opt]

print(f"Checking {job}...")

# Run check script
runs = { "env_fail": [], "timed_out": [], "failed_unknown": [], "slurm_sync_failed": [], "no_primary_data": [], "clusterless": [], "success": [], "unaccounted": []}
summary = {"env_fail": [], "timed_out": [], "failed_unknown": [], "slurm_sync_failed": [], "no_primary_data": [], "clusterless": [], "success": [], "unaccounted": []}

print("\nAnalyzing log output files...\n")
directories = os.listdir(f"{PATH}/{job}")

for filename in directories:
    if filename == "check_out":
        continue
    
    path = f"{PATH}/{job}/{filename}"
    f = open(path, "r")
    file = [line for line in f]
    file_str = "".join(file)
    f.close()

    # Checking status in order of log appearance in .out file:
    if "EnvironmentLocationNotFound:" in file[2]:   
        runs["env_fail"].append(extract_cell_type(file))
        summary["env_fail"].append(int(filename[19:-4]))

    elif "DUE TO TIME LIMIT ***" in file[-1]:       
        runs["timed_out"].append(extract_cell_type(file))
        summary["timed_out"].append(int(filename[19:-4]))

    elif "Failed to preprocess" in file_str:        
        if "Unable to synchronously open file" in file_str:        
            runs["slurm_sync_failed"].append(extract_cell_type(file))
            summary["slurm_sync_failed"].append(int(filename[19:-4]))
        else:
            runs["failed_unknown"].append(extract_cell_type(file))
            summary["failed_unknown"].append(int(filename[19:-4]))

    elif "No 'primary data' is contained in this cell-type" in file_str:        
        runs["no_primary_data"].append(extract_cell_type(file))
        summary["no_primary_data"].append(int(filename[19:-4]))

    elif "No clusters detected" in file_str:        
        runs["clusterless"].append(extract_cell_type(file))
        summary["clusterless"].append(int(filename[19:-4]))
        
    elif "Successfully ran ARACNe" in file_str: 
        runs["success"].append(extract_cell_type(file)) # Check that the processing was successful and completed for this cell-type
        summary["success"].append(int(filename[19:-4]))
        
    else:                                           
        runs["unaccounted"].append(extract_cell_type(file))
        summary["unaccounted"].append(int(filename[19:-4]))

# Unable to synchronously open file
# __conda_exe: command not found

print(f"# successfully preprocessed: {len(runs['success'])}")
print(f"# -- failed (clusterless): {len(runs['clusterless'])} \t\t -> Normal Behavior")
print(f"# -- failed (no primary data): {len(runs['no_primary_data'])} \t -> Normal Behavior")
print(f"# -- failed (SLURM sync): {len(runs['slurm_sync_failed'])} \t\t -> Please Investigate")
print(f"# -- failed (env-failure): {len(runs['env_fail'])} \t\t -> Please Investigate")
print(f"# -- failed (timed-out): {len(runs['timed_out'])} \t\t -> Please Investigate")
print(f"# -- failed (unknown): {len(runs['failed_unknown'])} \t\t -> Please Investigate")
print(f"# >> unaccounted-error: {len(runs['unaccounted'])} \t\t -> Please Investigate")

print("\nGenerating survey files...")

check_out_dir = f"{PATH}/{job}/check_out"
if not os.path.exists(check_out_dir):
    os.makedirs(check_out_dir) # Create the directory

for key in runs.keys():
    runs[key].sort() # Sort file directories
    out_file = open(f"{check_out_dir}/{key}.txt", "w")
    for directory in runs[key]:
        out_file.write(f"{directory}\n")
    out_file.close()

failed_out_file = open(f"{check_out_dir}/failed.txt", "w")
run_failed = runs["slurm_sync_failed"] + runs["failed_unknown"]
run_failed.sort()
for directory in run_failed:
    failed_out_file.write(f"{directory}\n")
failed_out_file.close()

# Below we write a separate file with information on which log file (rather than cell type) experienced issues
# This is mainly for diagnostic/debugging-use
sum_out_file = open(f"{check_out_dir}/SUMMARY.txt", "w")
sum_out_file.write(f"Below are the indices of the incomplete cell-types for {job[10:]}:\n\n")
for key in summary.keys():
    out = summary[key]
    out.sort()
    sum_out_file.write(f"{key}: {out}\n")
    
out = summary["slurm_sync_failed"] + summary["failed_unknown"]
out.sort()
sum_out_file.write(f"\noverall_failed: {out}\n")
    
out_file.close()

print(f"\nFiles were succesfully generated!")
print(f"Find them at: {PATH[:-10]}/slurm_out/{job}/check_out\n")