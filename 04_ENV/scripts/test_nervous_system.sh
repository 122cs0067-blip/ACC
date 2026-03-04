#!/bin/bash
# Combined IPC Test: Oracle + Student Handshake

echo "================================================"
echo " ACC Nervous System Test: Phase 1, Day 3"
echo "================================================"

# Step 1: Clean up any existing shared memory
echo ""
echo " Cleaning up any existing shared memory..."
python3 -c "from multiprocessing import shared_memory; shm = shared_memory.SharedMemory(name='acc_bridge'); shm.close(); shm.unlink()" 2>/dev/null || echo "   (No existing memory to clean)"

# Step 2: Start Oracle in background
echo ""
echo " Starting ORACLE (Host/CPU) in background..."
cd /home/cse-sdpl/research/ACC
conda run -n acc_env python 04_ENV/scripts/test_ipc_host.py &
ORACLE_PID=$!
echo "   Oracle PID: $ORACLE_PID"

# Wait for Oracle to initialize
sleep 3

# Step 3: Launch Student in Docker
echo ""
echo " Launching STUDENT (Docker/GPU)..."
sudo docker run --gpus all --ipc=host --rm \
  -v /home/cse-sdpl/research/ACC/02_SRC:/app/src \
  -v /home/cse-sdpl/research/ACC/04_ENV/scripts:/app/scripts \
  acc_student_gpu \
  python /app/scripts/test_ipc_student.py

# Step 4: Cleanup
echo ""
echo " Stopping Oracle..."
kill $ORACLE_PID 2>/dev/null
wait $ORACLE_PID 2>/dev/null

echo ""
echo "================================================"
echo " Test Complete"
echo "================================================"
