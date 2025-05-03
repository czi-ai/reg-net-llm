#!/bin/bash
#SBATCH --job-name=PRETRAIN
#SBATCH --output=./slurm_train_out/array_job_%A.out
#SBATCH --time=1-00:00:00
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
mamba activate scllm
# mamba activate scllm_2

echo
echo "Initiating GLM training..."
echo

srun python /hpc/mydata/leo.dupire/GLM/scGraphLLM/scGraphLLM/run_training.py --config="$CONFIG_NAME" --mode="train" --name="$RUN_NAME"

##########################################################################################
#################################### EXAMPLE COMMANDS ####################################
##########################################################################################

: <<'END_COMMENT'

sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_4096" --run-name "PRETRAIN [CLS, 12Layer, both_mask:15%]"

sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_3L_4096" --run-name "RUN [CLS, 3Layer]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_6L_4096" --run-name "RUN [CLS, 6Layer]"

sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_1DIFF_4096" --run-name "RUN [CLS, 12Layer, 1DIFF]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_2DIFF_A_4096" --run-name "RUN [CLS, 12Layer, 2DIFF:0,1]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_2DIFF_B_4096" --run-name "RUN [CLS, 12Layer, 2DIFF:0,6]"

sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_4096" --run-name "RUN [CLS, 12Layer, both_mask:100%]"
sbatch /hpc/mydata/leo.dupire/GLM/scGraphLLM/run.sh --config "graph_kernel_attn_1DIFF_4096" --run-name "RUN [CLS, 12Layer, 1DIFF, both_mask:100%]"

END_COMMENT