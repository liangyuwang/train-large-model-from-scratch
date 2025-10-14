#!/bin/bash
# ================================
# SSH-Based Multi-Node Launch Script
# ================================
#
# This script launches training on multiple nodes via SSH.
# Useful when you don't have SLURM or other job schedulers.
#
# Prerequisites:
#   1. SSH access to all nodes without password (use ssh-keygen)
#   2. Same code directory on all nodes (use rsync or NFS)
#   3. Same conda/virtual environment on all nodes
#
# Usage:
#   bash scripts/launch_multinode_ssh.sh
#

# ========================================
# Configuration - EDIT THIS SECTION
# ========================================

# List of node hostnames or IPs
# First node is master
NODES=(
    "192.168.1.100"
    "192.168.1.101"
    # "192.168.1.102"
    # "192.168.1.103"
)

# Training script to run
MODEL_SCRIPT="scripts/debug_gpt_0.25b/pretrain.sh"
# MODEL_SCRIPT="scripts/debug_gpt_0.3b_a0.17b/pretrain.sh"

# Network configuration
MASTER_PORT=29500

# Path to your project directory on all nodes
# If using NFS, this should be the same path on all nodes
PROJECT_DIR="$(pwd)"

# Optional: Conda environment name
CONDA_ENV=""  # Leave empty if not using conda
# CONDA_ENV="pytorch"

# Optional: Activate virtual environment command
VENV_ACTIVATE=""  # Leave empty if not using venv
# VENV_ACTIVATE="source ~/venv/bin/activate"

# ========================================
# End Configuration
# ========================================

NUM_NODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}

echo "=================================================="
echo "Multi-Node Training Launch"
echo "=================================================="
echo "Number of nodes: $NUM_NODES"
echo "Master node: $MASTER_ADDR"
echo "Worker nodes: ${NODES[@]:1}"
echo "Training script: $MODEL_SCRIPT"
echo "Project directory: $PROJECT_DIR"
echo "=================================================="

# Verify SSH access to all nodes
echo "Verifying SSH access to all nodes..."
for NODE in "${NODES[@]}"; do
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$NODE" echo "OK" &>/dev/null; then
        echo "ERROR: Cannot SSH to $NODE"
        echo "Please set up passwordless SSH access:"
        echo "  ssh-keygen -t rsa"
        echo "  ssh-copy-id $NODE"
        exit 1
    fi
    echo "  ✓ $NODE"
done
echo "All nodes are accessible!"
echo "=================================================="

# Function to build activation command
build_activation_cmd() {
    local cmd=""
    if [ -n "$CONDA_ENV" ]; then
        cmd="conda activate $CONDA_ENV && "
    elif [ -n "$VENV_ACTIVATE" ]; then
        cmd="$VENV_ACTIVATE && "
    fi
    echo "$cmd"
}

# Function to launch training on a node
launch_on_node() {
    local NODE_IP=$1
    local NODE_RANK=$2
    local LOG_FILE=$3
    
    local ACTIVATION_CMD=$(build_activation_cmd)
    
    echo "Launching on Node $NODE_RANK ($NODE_IP)..."
    
    ssh "$NODE_IP" "cd $PROJECT_DIR && \
        $ACTIVATION_CMD \
        NUM_NODES=$NUM_NODES \
        NODE_RANK=$NODE_RANK \
        MASTER_ADDR=$MASTER_ADDR \
        MASTER_PORT=$MASTER_PORT \
        bash $MODEL_SCRIPT" > "$LOG_FILE" 2>&1 &
    
    echo "  → Log: $LOG_FILE"
}

# Create log directory
LOG_DIR="logs/multinode_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Launching training on all nodes..."
echo "Logs will be saved to: $LOG_DIR"
echo "=================================================="

# Launch on all nodes
PIDS=()
for i in "${!NODES[@]}"; do
    NODE_IP=${NODES[$i]}
    LOG_FILE="$LOG_DIR/node${i}_${NODE_IP}.log"
    
    launch_on_node "$NODE_IP" "$i" "$LOG_FILE"
    PIDS+=($!)
    
    # Small delay to ensure master starts first
    if [ $i -eq 0 ]; then
        echo "Waiting 5 seconds for master node to initialize..."
        sleep 5
    fi
done

echo "=================================================="
echo "Training launched on all nodes!"
echo "=================================================="
echo "Monitor logs with:"
echo "  tail -f $LOG_DIR/node*.log"
echo ""
echo "Or use tmux/screen on each node:"
for i in "${!NODES[@]}"; do
    echo "  ssh ${NODES[$i]}"
done
echo ""
echo "To stop all training:"
echo "  bash scripts/stop_multinode_ssh.sh"
echo "=================================================="

# Optional: Wait for all background jobs
# Uncomment if you want the script to wait until training completes
# echo "Waiting for training to complete..."
# for PID in "${PIDS[@]}"; do
#     wait $PID
# done
# echo "All training jobs completed!"

# Save PIDs to file for later cleanup
echo "${PIDS[@]}" > "$LOG_DIR/pids.txt"
echo "Nodes: ${NODES[@]}" >> "$LOG_DIR/pids.txt"

echo "Launch script finished. Training is running in background."

