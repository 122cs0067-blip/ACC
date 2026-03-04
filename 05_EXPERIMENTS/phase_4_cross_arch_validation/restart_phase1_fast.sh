#!/usr/bin/env bash
# =============================================================
# Phase 1 Fast-Mode Restart Script
# Run this AFTER killing the current campaign (after POPE is done)
#
# What this does:
#   - Skips POPE (already complete at 3001 samples)
#   - Runs MathVista (500), VQAv2 (500), ALFWorld (135)
#   - Uses 20-token teacher cap (applied in oracle_bridge.py)
#   - Saves to a new v3_fast results directory
#
# Expected runtime: ~2-3 hours total
# =============================================================

set -e

CONDA_ENV="acc_bench_env"
SCRIPT_DIR="/home/cse-sdpl/research/ACC/05_EXPERIMENTS/phase_4_cross_arch_validation"
RESULTS_DIR="/home/cse-sdpl/research/ACC/06_RESULTS/final_campaign_v3"   # same dir — POPE results already here
LOG_FILE="/home/cse-sdpl/research/ACC/01_DATA/final_campaign_phase1_v3.log" # append to SAME log

echo "" >> "$LOG_FILE"
echo "============================================" >> "$LOG_FILE"
echo " FAST-MODE RESTART — $(date)" >> "$LOG_FILE"
echo " Benchmarks: mathvista(500) + vqav2(500) + alfworld(135)" >> "$LOG_FILE"
echo " Teacher cap: 20 tokens" >> "$LOG_FILE"
echo "============================================" >> "$LOG_FILE"

echo "============================================="
echo " ACC Phase 1 "
echo " Benchmarks: mathvista(500) + vqav2(500) + alfworld(135)"
echo " Teacher cap: 20 tokens"
echo " Results dir: $RESULTS_DIR (same as v3 — POPE already there)"
echo " Log: $LOG_FILE (APPENDING to existing log)"
echo "============================================="

mkdir -p "$RESULTS_DIR/phase_1"

# POPE results are already in $RESULTS_DIR/phase_1/ — no copy needed.
echo "[STEP 1] POPE results already present in $RESULTS_DIR/phase_1/ — skipping copy."

echo "[STEP 2] Launching remaining benchmarks in fast mode..."

conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/final_campaign_runner.py" \
    --phase 1 \
    --models "qwen2.5-vl-3b" "phi-4-multimodal" "llava-v1.6-7b" \
    --benchmarks "mathvista" "vqav2" "alfworld" \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG_FILE"   # -a = APPEND to existing v3 log

echo "============================================="
echo " Phase 1 Complete!"
echo " Results: $RESULTS_DIR"
echo " Log: $LOG_FILE"
echo "============================================="
