#!/usr/bin/env python3
"""
Final VLM Benchmark Campaign Runner (Sequential Phases)
Location: 05_EXPERIMENTS/phase_4_cross_arch_validation/final_campaign_runner.py

Phase 1: ACC Fleet (Fixed Thresholds, Active Handoff)
Phase 2: Baseline & Ablation (Passive Monitoring, Entropy/OPERA/VISTA)
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "02_SRC"))
sys.path.append(str(PROJECT_ROOT / "05_EXPERIMENTS/phase_4_cross_arch_validation"))

from acc_core.system.deployment_config import ACC_VLM_FLEET, DATASET_SPLITS
from benchmark_loaders_vlm import load_vlm_benchmark
from wrappers.campaign_logger import CampaignLogger

# Import specialized agents
from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
from wrappers.wrapper_llava16_7b import LLaVA16StudentAgent

MODEL_AGENT_MAP = {
    "qwen2.5-vl-3b": QwenVLStudentAgent,
    "phi-4-multimodal": Phi4MultimodalStudentAgent,
    "llava-v1.6-7b": LLaVA16StudentAgent
}

def run_phase_1(model_names, benchmark_names, results_dir, num_samples_override=None):
    """ACC Fleet Deployment: Fixed lambda*, Active Teacher."""
    print("\n" + "="*60)
    print("=== STARTING PHASE 1: ACC FLEET LAUNCH ===")
    print("="*60)
    if num_samples_override:
        print(f"[FAST MODE] num_samples capped at {num_samples_override} per benchmark")
    
    results_dir = Path(results_dir) / "phase_1"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Track overall progress — use override if set
    total_samples = 0
    for b_name in benchmark_names:
        n = num_samples_override if num_samples_override else DATASET_SPLITS[b_name]["num_samples"]
        total_samples += n
    overall_total = len(model_names) * total_samples
    overall_pbar = tqdm(total=overall_total, desc="Global Progress", unit="sample")

    for m_name in model_names:
        if m_name not in ACC_VLM_FLEET:
            print(f"[WARN] Model {m_name} not found in deployment config. Skipping.")
            continue
            
        config = ACC_VLM_FLEET[m_name]
        lambda_star = config["lambda_star"]
        agent_cls = MODEL_AGENT_MAP.get(m_name)
        
        if not agent_cls:
            print(f"[WARN] No agent wrapper for {m_name}. Skipping.")
            continue
            
        print(f"\n[PHASE 1] Loading Model: {m_name} (lambda*={lambda_star})")
        # Load agent with fixed threshold and teacher enabled
        agent = agent_cls(lambda_star=lambda_star, use_teacher=True)
        if agent.oracle:
            print(f"[PHASE 1] Pre-loading Oracle Teacher for {m_name}...")
            agent.oracle.load_teacher()
        
        for b_name in benchmark_names:
            if b_name not in DATASET_SPLITS:
                print(f"[WARN] Benchmark {b_name} not found in config. Skipping.")
                continue
            
            # Use override if provided, otherwise use config default
            n_samples = num_samples_override if num_samples_override else DATASET_SPLITS[b_name]["num_samples"]
            split = DATASET_SPLITS[b_name]
            print(f"  -> Running Benchmark: {b_name} (N={n_samples})")
            
            samples = load_vlm_benchmark(b_name, Path(split["path"]), num_samples=n_samples)

            
            output_file = results_dir / f"acc_results_{m_name}_{b_name}.jsonl"
            
            # Resume logic: Check how many samples are already done
            done_ids = set()
            if output_file.exists():
                with open(output_file, "r") as r:
                    for line in r:
                        try:
                            data = json.loads(line)
                            done_ids.add(data["sample_id"])
                        except:
                            pass
            
            if done_ids:
                print(f"  [RESUME] Found {len(done_ids)} existing results. Skipping ahead...")

            with open(output_file, "a") as f:
                for i, sample in enumerate(tqdm(samples, desc=f" {m_name} : {b_name}", leave=False)):
                    if str(sample.sample_id) in done_ids:
                        overall_pbar.update(1)
                        continue
                        
                    start_t = time.time()
                    
                    # Run inference with ACC protection
                    try:
                        res_dict = agent.run_text(
                            prompt=sample.question, 
                            image=sample.image,
                            task_id=sample.sample_id
                        )
                        response = res_dict["generated_text"]
                        chasm_detected = bool(res_dict["dcdr"])
                    except Exception as e:
                        import traceback
                        print(f"\n[ERROR] Failed sample {sample.sample_id}:")
                        traceback.print_exc()
                        response, chasm_detected = "ERROR", False
                    
                    latency = time.time() - start_t
                    
                    # Record result
                    entry = {
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "ground_truth": sample.ground_truth,
                        "prediction": response,
                        "acc_active": chasm_detected,
                        "latency_sec": latency,
                        "efficiency": res_dict.get("efficiency", 1.0),
                        "drift_scores": res_dict.get("drift_scores", []),
                        "interventions": res_dict.get("interventions", 0),
                        "metadata": sample.metadata
                    }
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    
                    # Explicit unbuffered print for log visibility
                    print(f"  [PROGRESS] {m_name} : {b_name} | Sample {i+1}/{n_samples} | ID: {sample.sample_id} | Handoff: {chasm_detected} | Latency: {latency:.2f}s", flush=True)
                    
                    overall_pbar.update(1)
            
        # Unload model to free VRAM for next one
        print(f"[PHASE 1] Unloading Model: {m_name}")
        agent.unload_model()
        del agent
        torch.cuda.empty_cache()
        time.sleep(5) # Cooldown

    overall_pbar.close()
    print("\n" + "="*60)
    print("=== PHASE 1 COMPLETE ===")
    print("="*60)

def run_phase_2(model_names, benchmark_names, baselines, results_dir):
    """Baseline & Ablation Sweep: Passive SOTA metrics."""
    print("\n" + "="*60)
    print("=== STARTING PHASE 2: BASELINE & ABLATION SWEEP ===")
    print("="*60)
    # To be implemented following Phase 1 success
    print("[NOTE] Phase 2 logic will follow the completion of the core ACC fleet launch.")
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACC Final Campaign Runner")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True, 
                        help="1 for ACC Fleet, 2 for Baselines")
    parser.add_argument("--models", nargs="+", default=["qwen2.5-vl-3b", "phi-4-multimodal", "llava-v1.6-7b"],
                        help="Models to run")
    parser.add_argument("--benchmarks", nargs="+", default=["pope", "vqav2", "mathvista", "alfworld"],
                        help="Benchmarks to run")
    parser.add_argument("--results_dir", type=str, default="/home/cse-sdpl/research/ACC/06_RESULTS/final_campaign",
                        help="Root directory for results")
    parser.add_argument("--num_samples", type=int, default=None, help="Force specific number of samples per benchmark")
    parser.add_argument("--test_mode", action="store_true", help="Run only 1 sample per benchmark (shorthand for --num_samples 1)")

    args = parser.parse_args()

    # Apply sample overrides
    limit = args.num_samples
    if args.test_mode:
        limit = 1
    
    if limit is not None:
        print(f"[OVERRIDE] Reducing samples to {limit} per benchmark.")
        for b in args.benchmarks:
            if b in DATASET_SPLITS:
                DATASET_SPLITS[b]["num_samples"] = limit

    if args.phase == 1:
        run_phase_1(args.models, args.benchmarks, args.results_dir,
                    num_samples_override=limit)
    else:
        # Baselines: Any4 (Naive), Uncertainty (Entropy), Attention (OPERA), Vision (VISTA)
        run_phase_2(args.models, args.benchmarks, ["any4", "entropy", "opera", "vista"], args.results_dir)
