#!/bin/bash
# Phase 1, Day 4: Launch the Complete ACC Agent System
# This runs Oracle + Student with the actual Llama-3 model

echo "================================================"
echo " ACC Agent System - Phase 1, Day 4"
echo "================================================"
echo ""
echo "This will:"
echo "  1. Start Oracle (CPU) monitoring system"
echo "  2. Launch Student (GPU) with Llama-3-8B-Instruct"
echo "  3. Stream token generation with real-time telemetry"
echo ""

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN not set!"
    echo ""
    echo "To download Llama-3, you need a Hugging Face token:"
    echo "  1. Go to https://huggingface.co/settings/tokens"
    echo "  2. Create a token (if needed)"
    echo "  3. Accept Meta's Llama-3 license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
    echo ""
    read -p "Enter your HF token (or press Ctrl+C to cancel): " HF_TOKEN
    export HF_TOKEN
fi

# Clean up any existing shared memory
python3 -c "from multiprocessing import shared_memory; shm = shared_memory.SharedMemory(name='acc_bridge'); shm.close(); shm.unlink()" 2>/dev/null

# Change to project directory
cd /home/cse-sdpl/research/ACC

echo ""
echo " Starting Oracle (Host/CPU) in background..."
# Start Oracle in background with output to log
(eval "$(conda shell.bash hook)" 2>/dev/null && conda activate acc_env 2>/dev/null && python 04_ENV/scripts/test_ipc_host.py) > /tmp/acc_oracle.log 2>&1 &
ORACLE_PID=$!
echo "   Oracle PID: $ORACLE_PID"
echo "   Oracle log: /tmp/acc_oracle.log"

# Wait for Oracle to initialize
echo "   Waiting for Oracle to initialize..."
sleep 3

# Monitor Oracle log in background
tail -f /tmp/acc_oracle.log &
TAIL_PID=$!

echo ""
echo " Launching Student with Llama-3-8B-Instruct..."
echo "   (First run will download ~5GB model)"
echo ""

# Run Student in Docker with model
sudo docker run --gpus all --ipc=host --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /home/cse-sdpl/research/ACC/02_SRC:/app/src \
  -v /home/cse-sdpl/research/ACC/01_DATA:/app/data \
  acc_student_gpu \
  python /app/src/wrappers/student_gpu_agent.py

# Cleanup
echo ""
echo " Stopping Oracle..."
kill $TAIL_PID 2>/dev/null
kill $ORACLE_PID 2>/dev/null
wait $ORACLE_PID 2>/dev/null

echo ""
echo "================================================"
echo " Session Complete"
echo "================================================"
