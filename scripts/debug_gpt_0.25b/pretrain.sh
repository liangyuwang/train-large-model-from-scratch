#!/bin/bash
# ================================
# Torch Distributed Training Script
# ================================
# 
# For multi-node training, set these environment variables:
#   NUM_NODES: number of nodes (default: 1)
#   NUM_GPUS: number of GPUs per node (default: 8)
#   NODE_RANK: rank of this node, 0 for master (default: 0)
#   MASTER_ADDR: IP address of the master node (default: localhost)
#   MASTER_PORT: port for communication (default: 29500)
#
# Example for 2 nodes:
#   Node 0 (master, IP: 192.168.1.100):
#     NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 bash scripts/debug_gpt_0.25b/pretrain.sh
#   Node 1:
#     NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash scripts/debug_gpt_0.25b/pretrain.sh
#

# Multi-node configuration (can be overridden by environment variables)
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
B=${B:-8}

echo "==================================="
echo "Distributed Training Configuration"
echo "==================================="
echo "NUM_NODES:    $NUM_NODES"
echo "NUM_GPUS_PER_NODE:     $NUM_GPUS"
echo "NODE_RANK:    $NODE_RANK"
echo "MASTER_ADDR:  $MASTER_ADDR"
echo "MASTER_PORT:  $MASTER_PORT"
echo "BATCH_SIZE_PER_DEVICE:   $B"
echo "==================================="

DISTRIBUTED_ARGS="\
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
"

EXP_NAME="debug_gpt_0.25b"
TRAINING_ARGS="\
  --exp_name $EXP_NAME \
  --seed 1337 \
  --dataset_path ... \
  --use_mock_data \
  --mock_data_num_samples 12800 \
  --log_dir ./log \
  --total_batch_size 2097152 \
  --B $B \
  --T 4096 \
  --max_lr 6e-4 \
  --min_lr 6e-5 \
  --weight_decay 0.1 \
  --grad_clip_value 1.0 \
  --warmup_steps 2000 \
  --max_epochs 1 \
  --debug \
  --do_save \
  --save_every_steps 500 \
  --use_compile \
"

MODEL_ARGS="\
  --block_size 4096 \
  --vocab_size 50304 \
  --num_layer 12 \
  --num_attention_heads 32 \
  --num_key_value_heads 4 \
  --hidden_size 1024 \
  --intermediate_size 4096 \
  --tied_lm_head \
  --dropout 0.0 \
"

torchrun $DISTRIBUTED_ARGS train.py $TRAINING_ARGS $MODEL_ARGS
