import os

def extract_cell_type(file):
    return file[0][:-4].split()[-1]

# CHANGE TO YOUR DIRECTORY !
PATH = "/burg/pmg/users/ld3154/scGraphLLM/scripts/slurm_out/slurm_out"
dirs = os.listdir(PATH)

job = None
# job = "array_job_786994"
if job == None:
    job_list = list(set([dir[:16] for dir in dirs]))
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

runs = { "env_fail": [], "success": [], "clusterless": [], "timed_out": [], "failed": [], "unaccounted": []}
failure_summary = {"env_fail": [], "timed_out": [], "failed": [], "unaccounted": []}
print("\nAnalyzing log output files...\n")
for filename in dirs:
    if job not in filename: # Not the right job
        continue

    path = f"{PATH}/{filename}"
    f = open(path, "r")
    file = [line for line in f]
    f.close()

    # Checking status in order of log appearance in .out file:
    if "EnvironmentLocationNotFound:" in file[2]:   
        runs["env_fail"].append(extract_cell_type(file))
        failure_summary["env_fail"].append(int(filename[17:-4]))
    elif "Failed to preprocess" in file[-5]:        
        runs["failed"].append(extract_cell_type(file))
        failure_summary["failed"].append(int(filename[17:-4]))
    elif "Successfully ran ARACNe for" in file[-2]: 
        runs["success"].append(extract_cell_type(file)) # Check that the processing was successful and completed for this cell-type
    elif "No clusters detected" in file[-2]:        
        runs["clusterless"].append(extract_cell_type(file))
    elif "DUE TO TIME LIMIT ***" in file[-1]:       
        runs["timed_out"].append(extract_cell_type(file))
        failure_summary["timed_out"].append(int(filename[17:-4]))
    else:                                           
        runs["unaccounted"].append(extract_cell_type(file))
        failure_summary["unaccounted"].append(int(filename[17:-4]))

print(f"# env-failure cell-types: {len(runs['env_fail'])}")
print(f"# failed cell-types: {len(runs['failed'])}")
print(f"# successfully preprocessed cell-types: {len(runs['success'])}")
print(f"# clusterless cell-types: {len(runs['clusterless'])}")
print(f"# timed-out cell-types: {len(runs['timed_out'])}")
print(f"# unaccounted-error cell-types: {len(runs['unaccounted'])}")

print("\nGenerating survey files...")

check_out_dir = f"{PATH}/../check_out_{job[10:]}"
if not os.path.exists(check_out_dir):
    os.makedirs(check_out_dir) # Create the directory

    for key in runs.keys():
        runs[key].sort() # Sort file directories
        out_file = open(f"{check_out_dir}/{key}_{job[10:]}.txt", "w")
        for directory in runs[key]:
            out_file.write(f"{directory}\n")
        out_file.close()

    # Below we write a separate file with information on which log file (rather than cell type) experienced issues
    # This is mainly for diagnostic/debugging-use
    fail_sum_out_file = open(f"{check_out_dir}/failure_summary_{job[10:]}.txt", "w")
    fail_sum_out_file.write(f"Below are the indices of the incomplete cell-types for {job[10:]}:\n\n")
    for key in failure_summary.keys():
        fail_sum_out_file.write(f"{key}: {failure_summary[key]}\n")
    out_file.close()

    print(f"\nFiles were succesfully generated!")
    print(f"Find them at: {PATH}/../\n")

else:
    print(f"{job[10:]} directory already exists! Aborted overwrite.\n")