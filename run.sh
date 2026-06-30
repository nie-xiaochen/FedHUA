#!/usr/bin/env bash
set -euo pipefail

# Example configuration corresponding to the FedHUA paper settings.
dataset="fashionmnist"
partition="noniid"
model="simplecnn-mnist"
clients=10
sample_fraction=1.0
local_iterations=100
warmup_rounds=50
total_rounds=70
beta=0.1

# Paper-reported query-generation settings.
base_budget=256
generation_steps=500
lr_g=0.01
alpha=1.0
budget_min_scale=0.5
budget_max_scale=2.0
lambda_u=0.3
lambda_b=0.1

# Query-retention thresholds and retry limit. Override to match the experiment protocol.
tau_g=0.70
tau_b=0.30
max_query_attempts=10

log_dir="./log/${dataset}_${partition}_${model}_beta${beta}_r${total_rounds}_it${local_iterations}_c${clients}_p${sample_fraction}"
mkdir -p "$log_dir"

python -u fedhua.py \
  --dataset "$dataset" --gpu "0" --partition "$partition" --model "$model" \
  --n_parties "$clients" --sample_fraction "$sample_fraction" \
  --num_local_iterations "$local_iterations" --beta "$beta" \
  --warmup_rounds "$warmup_rounds" --comm_round "$total_rounds" \
  --base_budget "$base_budget" --generation_steps "$generation_steps" --lr_g "$lr_g" \
  --alpha "$alpha" --budget_min_scale "$budget_min_scale" --budget_max_scale "$budget_max_scale" \
  --lambda_u "$lambda_u" --lambda_b "$lambda_b" \
  --tau_g "$tau_g" --tau_b "$tau_b" --max_query_attempts "$max_query_attempts" \
  --save_model 2>&1 | tee "$log_dir/fedhua.log"
