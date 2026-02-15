# Multi-Node Training Guide

This guide explains how to run distributed training across multiple nodes using the provided scripts.

## Quick Start

### Single Node (Default)

Simply run the script without any environment variables:

```bash
bash scripts/debug_gpt_0.25b/pretrain.sh
```

This will train on a single node with 8 GPUs (default).

### Multi-Node Training

#### Prerequisites

1. **Network Setup**: Ensure all nodes can communicate with each other
2. **Shared Storage** (recommended): All nodes should have access to the same checkpoint directory
3. **Same Environment**: All nodes should have the same code, dependencies, and environment

#### Example: 2-Node Training

Assume you have 2 nodes:
- Node 0 (Master): IP `192.168.1.100`, 8 GPUs
- Node 1 (Worker): IP `192.168.1.101`, 8 GPUs

**On Node 0 (Master):**
```bash
NUM_NODES=2 \
NODE_RANK=0 \
MASTER_ADDR=192.168.1.100 \
MASTER_PORT=29500 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

**On Node 1 (Worker):**
```bash
NUM_NODES=2 \
NODE_RANK=1 \
MASTER_ADDR=192.168.1.100 \
MASTER_PORT=29500 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

#### Example: 4-Node Training

For 4 nodes with IPs `192.168.1.100-103`:

**Node 0 (Master, IP: 192.168.1.100):**
```bash
NUM_NODES=4 NODE_RANK=0 MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

**Node 1 (IP: 192.168.1.101):**
```bash
NUM_NODES=4 NODE_RANK=1 MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

**Node 2 (IP: 192.168.1.102):**
```bash
NUM_NODES=4 NODE_RANK=2 MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

**Node 3 (IP: 192.168.1.103):**
```bash
NUM_NODES=4 NODE_RANK=3 MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

## Environment Variables

All scripts support the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NUM_NODES` | Total number of nodes | `1` |
| `NUM_GPUS` | Number of GPUs per node | `8` |
| `NODE_RANK` | Rank of this node (0 for master) | `0` |
| `MASTER_ADDR` | IP address of the master node | `localhost` |
| `MASTER_PORT` | Port for communication | `29500` |
| `B` | Micro batch size per GPU | `8` |

## SLURM Cluster

For SLURM-based HPC clusters, you can create a SLURM script:

### Example SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=gpt_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Get master node hostname
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Export environment variables
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Run on each node
srun bash scripts/debug_gpt_0.25b/pretrain.sh
```

Submit with:
```bash
sbatch scripts/slurm_train.sh
```

## SSH-Based Multi-Node Launch

If you have SSH access to all nodes, you can use a launch script:

### Example Launch Script

Save as `scripts/launch_multinode.sh`:

```bash
#!/bin/bash

# Configuration
NODES=("192.168.1.100" "192.168.1.101" "192.168.1.102" "192.168.1.103")
NUM_NODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500
SCRIPT="scripts/debug_gpt_0.25b/pretrain.sh"

# Launch on all nodes
for i in "${!NODES[@]}"; do
  NODE_IP=${NODES[$i]}
  echo "Launching on Node $i ($NODE_IP)..."
  
  if [ $i -eq 0 ]; then
    # Master node - run in foreground
    NUM_NODES=$NUM_NODES \
    NODE_RANK=$i \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    bash $SCRIPT &
  else
    # Worker nodes - run via SSH
    ssh $NODE_IP "cd $(pwd) && \
      NUM_NODES=$NUM_NODES \
      NODE_RANK=$i \
      MASTER_ADDR=$MASTER_ADDR \
      MASTER_PORT=$MASTER_PORT \
      bash $SCRIPT" &
  fi
done

# Wait for all background jobs
wait
```

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check firewall settings
   - Ensure `MASTER_PORT` is open on all nodes
   - Verify `MASTER_ADDR` is reachable from all nodes

2. **NCCL Errors**
   ```bash
   # Check NCCL debug info
   export NCCL_DEBUG=INFO
   ```

3. **Inconsistent Checkpoints**
   - Ensure all nodes share the same `log_dir`
   - Use network file system (NFS) or shared storage

4. **Different Node Startup Times**
   - Increase timeout: `export NCCL_TIMEOUT=1800`
   - Check that all nodes start within reasonable time

### Verification Commands

Check network connectivity:
```bash
# On worker node, ping master
ping -c 3 192.168.1.100

# Test port connectivity
nc -zv 192.168.1.100 29500
```

Check GPU availability:
```bash
nvidia-smi
```

Check PyTorch distributed:
```bash
python -c "import torch; print(torch.distributed.is_nccl_available())"
```

## Performance Tips

1. **InfiniBand**: If available, use InfiniBand for faster inter-node communication
   ```bash
   export NCCL_IB_DISABLE=0
   export NCCL_NET_GDR_LEVEL=5
   ```

2. **Ethernet**: For Ethernet networks
   ```bash
   export NCCL_SOCKET_IFNAME=eth0  # or your network interface
   ```

3. **Network Optimization**
   ```bash
   export NCCL_IB_TIMEOUT=22
   export NCCL_IB_RETRY_CNT=7
   ```

## Scaling Considerations

When scaling to multiple nodes:

1. **Batch Size**: `total_batch_size` should be divisible by `(B × T × NUM_NODES × NUM_GPUS)`
2. **Learning Rate**: Consider adjusting learning rate when scaling (linear scaling rule)
3. **Warmup**: May need longer warmup with more GPUs
4. **I/O**: Ensure data loading doesn't become bottleneck

### Example: Scaling from 1 to 4 nodes

```bash
# 1 node, 8 GPUs: total_batch_size = 2097152
# 4 nodes, 32 GPUs: total_batch_size = 8388608 (4x)
# Learning rate: 6e-4 → 2.4e-3 (4x, linear scaling)
```

## Monitoring

Monitor training across nodes:

```bash
# On each node, check GPU utilization
watch -n 1 nvidia-smi

# Check training logs
tail -f log/*/log.txt

# Monitor network traffic
iftop -i eth0
```

## Contact

For issues or questions about multi-node training, check:
- PyTorch distributed documentation
- NCCL documentation
- Your cluster administrator (for HPC systems)

