#!/bin/bash
#SBATCH --job-name=PRETRAIN
#SBATCH --output=./slurm_train_out/array_job_%A.out
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=h100
#SBATCH -p gpu


#should not need this setting
#unset SLURM_MEM_PER_NODE
nvidia-smi

# Display all variables set by slurm
env | grep "^SLURM" | sort

# Print hostname job executed on.
echo
echo "My hostname is: $(hostname -s)"
echo

RUN_NAME=""
CONFIG_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-name)
      RUN_NAME="$2"
      shift
      shift
      ;;
    --config)
      CONFIG_NAME="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

module load mamba
module load cuda/12.1.1_530.30.02
# mamba activate scllm
mamba activate scllm_2

echo
echo "Initiating fine tuning..."
echo

srun python /home/mingxuan.zhang/scGraphLLM/scGraphLLM/run_fine_tuning.py --fine-tune-config="$CONFIG_NAME" --mode="train" --name="$RUN_NAME"  \
                                                                          --ckpt-path="/hpc/mydata/mingxuan.zhang/GLM/model_out/3L_model_dlr/checkpoints/epoch=0-step=34000.ckpt"
