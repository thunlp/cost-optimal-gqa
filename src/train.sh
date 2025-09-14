#!/bin/bash
#SBATCH --partition=TEST1_CYF
#SBATCH --account=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --job-name='cogqa'

echo "STARTING TRAIN"

args="$@"

for arg in "$@"; do
    eval "$arg"
done

# Arguments with default values
n_gpus=${n_gpus:-8}
exp_group=${exp_group:-"e1"}
proj_name=${proj_name:-"cogqa"}
model_name=${model_name:-"gpt"}
model_config=${model_config:-"gpt-4-256"}
train_config=${train_config:-"60m_8k"}
comment=${comment:-""}
# Some arguments need to be constructed
run_name="${exp_group}_${model_config}_${train_config}_${comment}"
model_config="./configs/model/${model_config}.json"
train_config="./configs/training/${train_config}.json"


# Build command
cmd="accelerate launch --config_file ./configs/accelerate/multigpu_config.yaml"
cmd+=" --num_processes=${n_gpus} --main_process_port 17848 train.py"

cmd+=" --run_name=${run_name}"
cmd+=" --proj_name=${proj_name}"

# Add command-line arguments to the cmd string
for arg in "$@"; do
    cmd+=" --$arg"
done

cmd+=" --model_config=${model_config}"
cmd+=" --train_config=${train_config}"

echo "======== Final command ========"
echo "$cmd" | tr ' ' '\n'
echo "==============================="

export WANDB_MODE=offline

$cmd
