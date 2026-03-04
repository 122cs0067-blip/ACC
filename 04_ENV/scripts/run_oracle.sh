#!/bin/bash
# Phase 1 Day 3: Nervous System Test
# Run this script - it will show you instructions for Terminal 2

echo "================================================"
echo " ACC Nervous System Test - Phase 1"
echo "================================================"
echo ""
echo "This test requires TWO terminal windows."
echo ""
echo "TERMINAL 1 (This one - Oracle/Host):"
echo "  1. Keep this window visible"
echo "  2. Press ENTER to start Oracle listener"
echo ""
read -p "Press ENTER to start Oracle..." 

# Clean up any existing shared memory
python3 -c "from multiprocessing import shared_memory; shm = shared_memory.SharedMemory(name='acc_bridge'); shm.close(); shm.unlink()" 2>/dev/null

echo ""
echo " ORACLE STARTING..."
echo "================================================"
cd /home/cse-sdpl/research/ACC

# Initialize conda if needed and activate acc_env
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate acc_env 2>/dev/null || echo "(Using base environment)"
python 04_ENV/scripts/test_ipc_host.py
