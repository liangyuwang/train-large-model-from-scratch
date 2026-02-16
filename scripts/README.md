# Training Scripts Guide

This directory contains training scripts and launch utilities for single-node and multi-node distributed training.

## Directory Structure

```
scripts/
├── README.md                      # This file
├── debug_gpt_0.25b/
│   └── pretrain.sh               # 0.25B model training script
├── debug_gpt_0.3b_a0.17b/
│   └── pretrain.sh               # 0.3B MoE model training script
```

## Quick Start

### 1. Single Node Training (Default)

Simply run any model script:

```bash
# Train 0.25B model
bash scripts/debug_gpt_0.25b/pretrain.sh

# Train 0.3B MoE model
bash scripts/debug_gpt_0.3b_a0.17b/pretrain.sh
```

**Default configuration:**
- 1 node
- 8 GPUs
- Runs on localhost

### 2. Multi-Node Training

#### Option A: Environment Variables (Recommended)

**Node 0 (Master):**
```bash
NUM_NODES=2 \
NODE_RANK=0 \
MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

**Node 1 (Worker):**
```bash
NUM_NODES=2 \
NODE_RANK=1 \
MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

## Environment Variables

All scripts support these environment variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `NUM_NODES` | Total number of nodes | `1` | `4` |
| `NUM_GPUS` | GPUs per node | `8` | `8` |
| `NODE_RANK` | Rank of this node (0=master) | `0` | `0`, `1`, `2`, ... |
| `MASTER_ADDR` | Master node IP/hostname | `localhost` | `192.168.1.100` |
| `MASTER_PORT` | Communication port | `29500` | `29500` |
| `B` | Micro batch size per GPU | `8` | `16` |
| `SEP_SIZE` | SEP group size (sequence-expert joint, `--sep_size`) | `1` | `2`, `4`, `8` |

Notes:
- Dense models (`--use_moe` disabled): SEP degenerates to pure SP.
- `WORLD_SIZE` must be divisible by `SEP_SIZE`.
- Sequence length `T` must be divisible by `SEP_SIZE`.

## Model Configurations

### debug_gpt_0.25b

**Architecture:**
- 12 layers
- 32 attention heads (4 KV heads - GQA)
- Hidden size: 1024
- FFN size: 4096
- Total parameters: ~250M

**Training config:**
- Batch size: 2M tokens
- Learning rate: 6e-4 → 6e-5
- Sequence length: 4096

**Use case:** Quick debugging, proof of concept

### debug_gpt_0.3b_a0.17b

**Architecture:**
- 12 layers
- 32 attention heads (4 KV heads - GQA)
- Hidden size: 768
- **MoE: 8 experts, 2 active per token**
- Expert FFN size: 768
- Total parameters: ~300M (active: ~170M)

**Training config:**
- Batch size: 2M tokens
- Learning rate: 6e-4 → 6e-5
- Sequence length: 4096

**Use case:** Testing MoE architecture, sparse models

## Customizing Scripts

### Create Your Own Training Script

1. Copy an existing script:
```bash
cp -r scripts/debug_gpt_0.25b scripts/my_model
```

2. Edit `scripts/my_model/pretrain.sh`:

```bash
# Change experiment name
EXP_NAME="my_model"

# Adjust training parameters
TRAINING_ARGS="\
  --exp_name $EXP_NAME \
  --total_batch_size 4194304 \     # 4M tokens
  --B 16 \                          # Larger micro batch
  --max_lr 3e-4 \                   # Different LR
  ...
"

# Adjust model architecture
MODEL_ARGS="\
  --num_layer 24 \                  # Deeper model
  --hidden_size 2048 \              # Larger hidden size
  ...
"
```

3. Run your script:
```bash
bash scripts/my_model/pretrain.sh
```

### Modify Batch Size for Different GPU Counts

The gradient accumulation is computed automatically:
```
grad_accum_steps = total_batch_size / (B × T × NUM_NODES × NUM_GPUS)
```

**Example: Scale from 8 to 32 GPUs**

Keep same effective batch size:
```bash
# 1 node, 8 GPUs
B=8  SEP_SIZE=1  total_batch_size=2097152

# 4 nodes, 32 GPUs - keep B=8, gradient accumulation reduces 4x
B=8  SEP_SIZE=4  total_batch_size=2097152

# Or increase batch size 4x
B=8  SEP_SIZE=4  total_batch_size=8388608
```

Or increase micro batch size:
```bash
# 4 nodes, 32 GPUs with larger micro batch
B=32  total_batch_size=8388608
```

## Common Use Cases

### Case 1: Quick Single-GPU Debug

```bash
NUM_GPUS=1 B=2 bash scripts/debug_gpt_0.25b/pretrain.sh
```

### Case 2: Full Node Training (8 GPUs)

```bash
bash scripts/debug_gpt_0.25b/pretrain.sh

# enable SEP (sequence-expert joint) parallelism on 2-way split
SEP_SIZE=2 bash scripts/debug_gpt_0.25b/pretrain.sh
```

### Case 3: Multi-Node Training (2 nodes, 16 GPUs)

**Node 0:**
```bash
NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=node0 SEP_SIZE=4 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

**Node 1:**
```bash
NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=node0 SEP_SIZE=4 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

## Monitoring Training

### View Logs

```bash
# Single node
tail -f log/*/log.txt

# Multi-node (SSH launcher)
tail -f logs/multinode_*/node*.log
```

### GPU Utilization

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Compact view
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'
```

### Training Metrics

Logs contain:
```
<step> train <loss>
<step> val <val_loss>
```

Extract training loss:
```bash
grep "train" log/*/log.txt | tail -20
```

### Network Traffic (Multi-Node)

```bash
# Install iftop if needed: sudo apt install iftop
sudo iftop -i eth0

# Or use nethogs
sudo nethogs eth0
```

## Checkpointing

### Checkpoint Files

Training saves:
```
log/<exp_name>_<config>/
├── 00000_model.pt          # Model weights
├── 00000_opt/              # Optimizer states (ZeRO-1 sharded)
├── 00000_meta.pt           # Metadata (step, RNG, etc.)
├── 05000_model.pt
├── 05000_opt/
├── 05000_meta.pt
└── log.txt
```

### Resume Training

Simply restart the same command. The trainer automatically:
- Finds latest checkpoint
- Restores model weights
- Restores optimizer states
- Restores RNG states
- Continues from correct step

**Important for multi-node:**
- Ensure checkpoint directory is accessible from all nodes (NFS/shared storage)
- Use same `exp_name` and configuration

## Troubleshooting

### Problem: Out of Memory

**Solutions:**
1. Reduce micro batch size: `B=4` or `B=2`
2. Reduce sequence length in `TRAINING_ARGS`: `--T 2048`
3. Enable gradient checkpointing (edit `train.py`)

### Problem: NCCL Timeout (Multi-Node)

**Solutions:**
```bash
# Increase timeout
export NCCL_TIMEOUT=1800

# Enable debug info
export NCCL_DEBUG=INFO

# Check network interface
export NCCL_SOCKET_IFNAME=eth0
```

### Problem: Slow Training

**Checks:**
1. MFU% in logs (should be >30%)
2. GPU utilization: `nvidia-smi` (should be >90%)
3. Network bandwidth (multi-node)

**Solutions:**
1. Increase micro batch size: `B=16`
2. Enable compilation: `--use_compile` in `TRAINING_ARGS`
3. Check data loading isn't bottleneck

### Problem: Different Results on Different Runs

Training is deterministic with same `--seed`. If results differ:
1. Check if seed is same
2. Verify same number of GPUs
3. Ensure same PyTorch version

## Performance Tips

### 1. Maximize Throughput

```bash
# Increase micro batch size
B=16  # or B=32 if memory allows

# Enable PyTorch compilation
--use_compile  # Add to TRAINING_ARGS
```

### 2. Multi-Node Optimization

```bash
# Use InfiniBand if available
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Otherwise, specify network interface
export NCCL_SOCKET_IFNAME=eth0
```

### 3. Large Batch Training

When using large batches:
```bash
# Scale learning rate linearly with batch size
# If 8 GPUs → 32 GPUs (4x):
max_lr: 6e-4 → 2.4e-3

# Increase warmup steps
warmup_steps: 2000 → 8000
```

## Reference

For more details, see:
- `../README.md` - Main project documentation
- `../train.py` - Training script source code

## Getting Help

If you encounter issues:
1. Check logs in `log/` directory
2. Enable debug output: `export NCCL_DEBUG=INFO`
3. Verify environment: `python -c "import torch; print(torch.cuda.is_available())"`
4. Review error messages carefully

Common log locations:
- Single node: `log/<exp_name>/log.txt`
- SLURM: `logs/train_<job_id>.out`
- SSH launcher: `logs/multinode_<timestamp>/node*.log`

