#!/bin/bash 
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --account=pmg
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --signal=SIGINT@1800
#SBATCH --job-name=run_training_scllm
#SBATCH --output=/burg/pmg/collab/scGraphLLM/data/slurm_logs/slurm-%j.out
# ^resources for slurm job; must have ntasks-per-node mtch devices in ptl
# signal sends a ctrl-c to stop the lightning process 30 min before end of allocation

set -e # exit on error
homdir=$PWD
if [ ! -d "/pmglocal/scGraphLLM/modeldata/pilotdata_cache" ]; then 
    echo "Folder does not exist. Creating"
    mkdir -p /pmglocal/scGraphLLM/modeldata/
    cp /burg/pmg/collab/scGraphLLM/data/pilotdata_cache.tgz  /pmglocal/scGraphLLM/modeldata/
    cd /pmglocal/scGraphLLM/modeldata/
    tar -xzf pilotdata_cache.tgz
    cd $homdir
else
    echo "Folder exists. Skipping"
fi

mkdir -p  /pmglocal/$USER/model_out ## if we have not been on this node before, create a model_out folder
rng=`echo $RANDOM | md5sum | head -c 12` ## generate a random version number to avoid overwriting
## create and set tmpdir to avoid interaction with other jobs
mkdir -p  /pmglocal/$USER/tmp/$rng
export WANDB__SERVICE_WAIT=300 ## this is a wandb setting to wait 300 seconds before timing out, was having problems with wandb timing out 
export TMPDIR="/pmglocal/$USER/tmp/$rng"
export PL_RECONCILE_PROCESS=1 ## these enables lightning to log output from multiple process/
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#echo $SLURM_MEM_PER_CPU $SLURM_MEM_PER_GPU $SLURM_MEM_PER_NODE 
#unset SLURM_MEM_PER_GPU
/pmglocal/$USER/mambaforge/envs/scllm/bin/python scGraphLLM/run_training.py "$@" ## @ is the bash wildcard for arguments passed into this shell script 

## srun is requried for multi-gpu training. however, if you need to run this in an interactive session you should delete the srun command and run the python script directly.