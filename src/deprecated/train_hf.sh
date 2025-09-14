#!/bin/bash
#SBATCH --account=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

proj_name="ffn"
run_name="vanilla_ng"
# run_name="tie_kv_ng"
# run_name="mhf"

ffn_is_gated="0"
ffn_tie_kv="0"
use_mhf="0"
mhf_n_heads="1"
d_model="256"
mhf_dim_k="256"
mhf_dim_v="256"
n_train_iters="20000"
n_decay_iters="2000"
n_warmup_iters="1000"

data_path="/home/test/test07/data/slimpj-chunked"
export WANDB_MODE=offline

cmd="accelerate launch train_hf.py"
cmd+=" --ffn_tie_kv ${ffn_tie_kv}"
cmd+=" --ffn_is_gated ${ffn_is_gated}"
cmd+=" --use_mhf ${use_mhf}"
cmd+=" --mhf_n_heads ${mhf_n_heads}"
cmd+=" --mhf_dim_k ${mhf_dim_k}"
cmd+=" --mhf_dim_v ${mhf_dim_v}"
cmd+=" --n_layer 6"
cmd+=" --d_model 512"
cmd+=" --grad_accum_steps 16"
cmd+=" --batch_size 16"
cmd+=" --max_len 1024"

cmd+=" --n_train_iters ${n_train_iters}"
cmd+=" --n_drop_iters ${n_decay_iters}"
cmd+=" --n_warmup_iters ${n_warmup_iters}"
cmd+=" --run_name ${run_name}"
cmd+=" --project_name ${proj_name}"

# Data
cmd+=" --data_path ${data_path}"

echo "RUNNING: $cmd"

$cmd
