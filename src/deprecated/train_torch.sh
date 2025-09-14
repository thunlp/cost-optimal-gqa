#!/bin/bash
#SBATCH --account=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

# model_config="./configs/model/gpt-124m.json"
n_layer="6"
d_model="512"

# Attention
n_head="8"
dim_k="64"
dim_v="64"
att_q_norm="1"
att_k_norm="1"

# FFN
ffn_per_layer="1"
ffn_is_gated="0"
ffn_act_fn="silu"
ffn_tie_kv="0"
ffn_d_mid="2048"
use_mhf="0"
mhf_n_heads="1"
mhf_dim_k="1024"
mhf_dim_v="1024"
mhf_output_norm="1"
mhf_q_norm="1"
mhf_use_o_proj="1"
mhf_use_q_proj="1"

# LR
n_train_iters="20000"
n_drop_iters="2000"
n_warmup_iters="1000"

# Data
batch_size="16"
max_len="1024"
grad_accum_steps="16"

proj_name="kv-merging"
exp_group="exp4"
# run_name="${exp_group}_gpt-45m_mhf${use_mhf}_fdk${mhf_dim_k}_fdv${mhf_dim_v}_dm${ffn_d_mid}_tie${ffn_tie_kv}_wq${mhf_use_q_proj}_qnorm${mhf_q_norm}_wo${mhf_use_o_proj}_onorm${mhf_output_norm}"
run_name="${exp_group}_gpt-45m_gated-ffn_dff${ffn_d_mid}"
# run_name="gpt-45m"

data_name="slimpj"
data_path="/home/test/test07/data/slimpj-chunked"
output_dir="output/${run_name}"

num_workers="2"

export WANDB_MODE=offline

# Build command
cmd="torchrun --standalone --nproc_per_node=8 train_torch.py"
cmd+=" --num_workers ${num_workers}"
cmd+=" --output_dir ${output_dir}"
cmd+=" --run_name ${run_name}"
cmd+=" --project_name ${proj_name}"
cmd+=" --compile 0"

# Model
# cmd+=" --model_config ${model_config}"
cmd+=" --n_layer ${n_layer}"
cmd+=" --d_model ${d_model}"

# FFN
cmd+=" --ffn_per_layer ${ffn_per_layer}"
cmd+=" --ffn_act_fn ${ffn_act_fn}"
cmd+=" --ffn_tie_kv ${ffn_tie_kv}"
cmd+=" --ffn_is_gated ${ffn_is_gated}"
cmd+=" --ffn_d_mid ${ffn_d_mid}"
cmd+=" --use_mhf ${use_mhf}"
cmd+=" --mhf_n_heads ${mhf_n_heads}"
cmd+=" --mhf_dim_k ${mhf_dim_k}"
cmd+=" --mhf_dim_v ${mhf_dim_v}"
cmd+=" --mhf_q_norm ${mhf_q_norm}"
cmd+=" --mhf_use_q_proj ${mhf_use_q_proj}"
cmd+=" --mhf_use_o_proj ${mhf_use_o_proj}"
cmd+=" --mhf_output_norm ${mhf_output_norm}"

# Attention
cmd+=" --n_head ${n_head}"
cmd+=" --dim_k ${dim_k}"
cmd+=" --dim_v ${dim_v}"
cmd+=" --att_q_norm ${att_q_norm}"
cmd+=" --att_k_norm ${att_k_norm}"

# LR
cmd+=" --n_train_iters ${n_train_iters}"
cmd+=" --n_drop_iters ${n_drop_iters}"
cmd+=" --n_warmup_iters ${n_warmup_iters}"

# Data
cmd+=" --data_name ${data_name}"
cmd+=" --data_path ${data_path}"
cmd+=" --batch_size ${batch_size}"
cmd+=" --max_len ${max_len}"
cmd+=" --grad_accum_steps ${grad_accum_steps}"

echo "RUNNING: $cmd"

# Execute
$cmd
