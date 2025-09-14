#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16

args="$@"

for arg in "$@"; do
    eval "$arg"
done

# Arguments with default values
exp_group=${exp_group:-"deepspeed"}
proj_name=${proj_name:-"MyProj"}
model_config=${model_config:-"gpt-8-1024"}
train_config=${train_config:-"10b_bsz-512k"}
comment=${comment:-""}
# Some arguments need to be constructed
run_name="${exp_group}_${model_config}_${train_config}_${comment}"
model_config="./configs/model/${model_config}.json"
train_config="./configs/training/${train_config}.json"

log_file=${log_file:-"logs/${run_name}.log"}
master_addr=${master_addr:-"g35"}
# num_processes=${num_processes:-8}
gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-2}
machine_rank=${machine_rank:-0}
master_ip=${master_ip:-"g35"}
master_port=${master_port:-6603}
n_gpus=$(( $n_machines * $gpus_per_node ))

echo "[$machine_rank] START TIME: $(date)"
echo "========== Distributed Settings ============"
echo "[$machine_rank] Running on $n_machines machines, with $n_gpus GPUs in total"
echo "[$machine_rank] Master IP: $master_ip"
echo "[$machine_rank] Master Port: $master_port"
echo "============================================"

export LAUNCHER="accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    --main_process_ip $master_ip \
    --main_process_port $master_port \
    --machine_rank $machine_rank \
    --num_processes $n_gpus \
    --num_machines $n_machines \
    "

# Build command
PROGRAM="train.py"

batch_size=${batch_size:-64}
per_device_batch_size=$(( $batch_size / $n_gpus ))

PROGRAM+=" --proj_name=${proj_name}"
PROGRAM+=" --batch_size=${per_device_batch_size}"

# Add command-line arguments to the PROGRAM string
for arg in "$@"; do
    PROGRAM+=" --$arg"
done

PROGRAM+=" --run_name=${run_name}"
PROGRAM+=" --model_config=${model_config}"
PROGRAM+=" --train_config=${train_config}"

echo "======== Final command ========"
echo "$PROGRAM" | tr ' ' '\n'
echo "==============================="

export WANDB_MODE=offline

export CMD="$LAUNCHER $PROGRAM"

$CMD

echo "END TIME: $(date)"
