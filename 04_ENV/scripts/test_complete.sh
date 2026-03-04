#!/bin/bash
# Complete automated test of the Nervous System

echo "================================================"
echo " ACC Nervous System - Automated Test"
echo "================================================"

# Clean up any existing shared memory
python3 -c "from multiprocessing import shared_memory; shm = shared_memory.SharedMemory(name='acc_bridge'); shm.close(); shm.unlink()" 2>/dev/null

# Change to project directory
cd /home/cse-sdpl/research/ACC

echo ""
echo " Starting Oracle (Host/CPU) in background..."
# Activate conda and run Oracle in background
(eval "$(conda shell.bash hook)" 2>/dev/null && conda activate acc_env 2>/dev/null && python 04_ENV/scripts/test_ipc_host.py) &
ORACLE_PID=$!
echo "   Oracle PID: $ORACLE_PID"

# Wait for Oracle to initialize
echo "   Waiting for Oracle to initialize..."
sleep 3

echo ""
echo " Launching Student (Docker/GPU)..."
echo "   (You may be prompted for sudo password)"
echo ""

# Run Student in Docker
sudo docker run --gpus all --ipc=host --rm \
  -v /home/cse-sdpl/research/ACC/02_SRC:/app/src \
  -v /home/cse-sdpl/research/ACC/04_ENV/scripts:/app/scripts \
  acc_student_gpu \
  python /app/scripts/test_ipc_student.py

# Give Oracle time to process final messages
sleep 2

echo ""
echo " Stopping Oracle..."
kill $ORACLE_PID 2>/dev/null
wait $ORACLE_PID 2>/dev/null

echo ""
echo "================================================"
echo " Test Complete"
echo "================================================"
echo ""
echo "Expected Results:"
echo "  • Oracle should have received Steps 1-9"
echo "  • Oracle should trigger intervention at Step 5"
echo "  • Student should detect intervention at Step 6"
echo ""
