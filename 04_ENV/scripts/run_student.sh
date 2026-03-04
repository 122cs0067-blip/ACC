#!/bin/bash
# Student side of the Nervous System test
# Run this in a SECOND terminal after Oracle is listening

echo "================================================"
echo " STUDENT (Docker/GPU) Starting..."
echo "================================================"
echo ""
echo "Note: You may be prompted for your sudo password"
echo ""

cd /home/cse-sdpl/research/ACC

sudo docker run --gpus all --ipc=host --rm \
  -v /home/cse-sdpl/research/ACC/02_SRC:/app/src \
  -v /home/cse-sdpl/research/ACC/04_ENV/scripts:/app/scripts \
  acc_student_gpu \
  python /app/scripts/test_ipc_student.py

echo ""
echo "================================================"
echo " Student Complete - Check Terminal 1 for Oracle messages"
echo "================================================"
