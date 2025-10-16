#!/bin/bash
#SBATCH --job-name=gpt_pretrain
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=1          # One task per node (torchrun handles the rest)
#SBATCH --gpus-per-node=8            # GPUs per node
#SBATCH --cpus-per-task=32           # CPUs per task
#SBATCH --time=48:00:00              # Maximum runtime
#SBATCH --partition=gpu              # Partition name (adjust to your cluster)
#SBATCH --output=logs/train_%j.out   # Standard output
#SBATCH --error=logs/train_%j.err    # Standard error
#SBATCH --exclusive                  # Exclusive node access

# ================================
# SLURM Multi-Node Training Script
# ================================
#
# Usage:
#   sbatch scripts/slurm_multinode.sh
#
# To customize:
#   1. Adjust SBATCH directives above
#   2. Set MODEL_SCRIPT below to your desired model
#   3. Modify training parameters if needed
#

# Create log directory
mkdir -p logs

# Print job info
echo "=================================================="
echo "SLURM Job Information"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "Start Time: $(date)"
echo "=================================================="

# Get master node address (first node in the allocation)
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Get node rank from SLURM
NODE_RANK=$SLURM_NODEID

# Set number of GPUs per node
NUM_GPUS=${SLURM_GPUS_PER_NODE:-8}

echo "Distributed Training Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK"
echo "  NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "  NUM_GPUS: $NUM_GPUS"
echo "=================================================="

# Choose which model to train
MODEL_SCRIPT="scripts/debug_gpt_0.25b/pretrain.sh"
# MODEL_SCRIPT="scripts/debug_gpt_0.3b_a0.17b/pretrain.sh"

# Export environment variables for the training script
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS=$NUM_GPUS
export NODE_RANK=$NODE_RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Optional: NCCL settings for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0              # Set to 1 if no InfiniBand
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback and docker interfaces
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_RETRY_CNT=7

# Run the training script
echo "Starting training on node $NODE_RANK..."
srun --nodes=1 --ntasks=1 --exclusive bash $MODEL_SCRIPT

echo "=================================================="
echo "Training completed at $(date)"
echo "=================================================="

