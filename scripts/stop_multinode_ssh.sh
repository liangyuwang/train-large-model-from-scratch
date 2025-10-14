#!/bin/bash
# ================================
# Stop Multi-Node Training Script
# ================================
#
# This script stops all training processes on remote nodes
#
# Usage:
#   bash scripts/stop_multinode_ssh.sh
#

# ========================================
# Configuration - Should match launch script
# ========================================

# List of node hostnames or IPs (same as launch script)
NODES=(
    "192.168.1.100"
    "192.168.1.101"
    # "192.168.1.102"
    # "192.168.1.103"
)

# ========================================
# End Configuration
# ========================================

echo "=================================================="
echo "Stopping Multi-Node Training"
echo "=================================================="

# Function to kill training processes on a node
stop_on_node() {
    local NODE_IP=$1
    
    echo "Stopping training on $NODE_IP..."
    
    # Kill all python processes related to train.py
    ssh "$NODE_IP" "pkill -f 'python.*train.py' || true"
    
    # Also kill torchrun processes
    ssh "$NODE_IP" "pkill -f 'torchrun' || true"
    
    echo "  ✓ Stopped processes on $NODE_IP"
}

# Stop training on all nodes
for NODE in "${NODES[@]}"; do
    stop_on_node "$NODE" &
done

# Wait for all stop commands to complete
wait

echo "=================================================="
echo "All training processes stopped!"
echo "=================================================="

# Optionally, check if any processes are still running
echo "Verifying processes are stopped..."
for NODE in "${NODES[@]}"; do
    RUNNING=$(ssh "$NODE" "pgrep -f 'train.py' | wc -l")
    if [ "$RUNNING" -gt 0 ]; then
        echo "  ⚠ Warning: $RUNNING process(es) still running on $NODE"
    else
        echo "  ✓ No processes running on $NODE"
    fi
done

echo "=================================================="
echo "Done!"

