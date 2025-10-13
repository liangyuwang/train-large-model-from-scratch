#!/bin/bash
# ================================
# Torch Distributed Training Script
# ================================

NUM_NODES=1
NUM_GPUS=8
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
B=8

DISTRIBUTED_ARGS="\
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
"

TRAINING_ARGS="\
  --seed 1337 \
  --dataset_path  \
  --use_mock_data \
  --mock_data_num_samples 12800 \
  --log_dir ./log \
  --tokenizer_name gpt2 \
  --total_batch_size 2097152 \
  --B $B \
  --T 4096 \
  --shift 1 \
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
