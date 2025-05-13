#!/bin/bash
#SBATCH --job-name=Pretrain
#SBATCH --output=./slurm_train_out/array_job_%A.out
#SBATCH --time=3-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=16
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
mamba activate scllm
# mamba activate scllm_2

echo
echo "Initiating GLM training..."
echo

# if [[ -n "$NUM_DEVICES" && "${NUM_DEVICES,,}" != "null" ]]; then
#   NUM_DEVICES_ARG="--devices $NUM_DEVICES"
# else
#   NUM_DEVICES_ARG=""
# fi

NUM_GPUS=$SLURM_GPUS | awk -F',' '{print NF}'
echo $NUM_GPUS=$SLURM_GPUS
NUM_DEVICES_ARG="--devices $SLURM_GPUS"

# srun python /hpc/mydata/leo.dupire/GLM/scGraphLLM/scGraphLLM/run_training.py --config="$CONFIG_NAME" --mode="train" --name="$RUN_NAME" # $NUM_DEVICES_ARG
srun python /hpc/mydata/leo.dupire/GLM/scGraphLLM/scGraphLLM/run_training.py --config="$CONFIG_NAME" --mode="resume" --name="$RUN_NAME" --version "PRETRAIN [CLS, 6Layer, 0-2-4Diff, Q-mixing]:2025-05-12@22:24:48" --ckpt-file "epoch=0-step=1000.ckpt" # $NUM_DEVICES_ARG

##########################################################################################
#################################### EXAMPLE COMMANDS ####################################
##########################################################################################

: <<'END_COMMENT'

sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_3L_4096" --run-name "PRETRAIN [CLS, 3Layer, 3Diff, lr:5e-5, Q-mixing, Shuffled]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_3L_1DIFF_4096" --run-name "PRETRAIN [CLS, 3Layer, 1Diff, lr:5e-5, Q-mixing]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_6L_3DIFF_4096" --run-name "PRETRAIN [CLS, 6Layer, 0-2-4Diff, Q-mixing]"

graph_kernel_attn_6L_4096


sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_3L_4096_sc" --run-name "PRETRAIN Single Cell [CLS, 3Layer, expression_mask:15%, lr:5e-5, AdamW]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_6L_4096" --run-name "PRETRAIN [CLS, 6Layer, expression_mask:15%, lr:5e-5, AdamW]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_4096" --run-name "PRETRAIN [CLS, 12Layer, expression_mask:15%, lr:5e-5, AdamW]"

END_COMMENT