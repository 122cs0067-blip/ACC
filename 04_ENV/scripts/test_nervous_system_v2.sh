#!/bin/bash
# Enhanced IPC Test with visible Oracle output

echo "================================================"
echo "ACC Nervous System Test: Phase 1 
echo "================================================"

# Clean up any existing shared memory
echo ""
echo " Cleaning up any existing shared memory..."
python3 -c "from multiprocessing import shared_memory; shm = shared_memory.SharedMemory(name='acc_bridge'); shm.close(); shm.unlink()" 2>/dev/null || echo "   (No existing memory to clean)"

# Create a log file for Oracle output
ORACLE_LOG="/tmp/acc_oracle_test.log"
rm -f $ORACLE_LOG

# Start Oracle in background with output redirected to log
echo ""
echo " Starting ORACLE (Host/CPU) in background..."
cd /home/cse-sdpl/research/ACC
conda run -n acc_env python 04_ENV/scripts/test_ipc_host.py > $ORACLE_LOG 2>&1 &
ORACLE_PID=$!
echo "   Oracle PID: $ORACLE_PID"

# Wait for Oracle to initialize
sleep 3

# Monitor Oracle log in background
echo ""
echo " Oracle Output:"
tail -f $ORACLE_LOG &
TAIL_PID=$!

# Launch Student in Docker
echo ""
echo " Launching STUDENT (Docker/GPU)..."
sudo docker run --gpus all --ipc=host --rm \
  -v /home/cse-sdpl/research/ACC/02_SRC:/app/src \
  -v /home/cse-sdpl/research/ACC/04_ENV/scripts:/app/scripts \
  acc_student_gpu \
  python /app/scripts/test_ipc_student.py

# Give Oracle time to process final messages
sleep 2

# Cleanup
echo ""
echo " Stopping Oracle..."
kill $TAIL_PID 2>/dev/null
kill $ORACLE_PID 2>/dev/null
wait $ORACLE_PID 2>/dev/null

# Show final Oracle output
echo ""
echo "================================================"
echo " Full Oracle Log:"
echo "================================================"
cat $ORACLE_LOG

echo ""
echo "================================================"
echo " Test Complete"
echo "================================================"
