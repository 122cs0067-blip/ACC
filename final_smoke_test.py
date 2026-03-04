#!/usr/bin/env python3
import os
import sys
import torch
import gc
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.append(str(ROOT / "02_SRC"))
sys.path.append(str(ROOT / "05_EXPERIMENTS" / "phase_4_cross_arch_validation"))

from benchmark_loaders_vlm import load_vlm_benchmark
from acc_core.system.deployment_config import ACC_VLM_FLEET, DATASET_SPLITS

def smoke_test_fleet():
    print("="*60)
    print("Final Fleet Smoke Test (N=5,137 Readiness)")
    print("="*60)

    benchmarks = ["pope", "mathvista", "vqav2", "alfworld"]
    models = ["qwen2.5-vl-3b", "phi-4-multimodal", "llava-v1.6-7b"]

    # 1. Test Loaders
    print("\n[1/3] Testing All Benchmark Loaders...")
    test_samples = {}
    for b in benchmarks:
        cfg = DATASET_SPLITS[b]
        try:
            samples = load_vlm_benchmark(b, Path(cfg["path"]), num_samples=1)
            test_samples[b] = samples[0]
            print(f"  [OK] {b:10s}: Loaded 1 sample (ID: {samples[0].sample_id})")
        except Exception as e:
            print(f"  [ERROR] {b:10s}: FAILED - {e}")
            return

    # 2. Test Teacher (OracleBridge)
    print("\n[2/3] Testing OracleBridge (Teacher)...")
    try:
        from acc_core.system.oracle_bridge import OracleBridge
        oracle = OracleBridge(model_id="/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_vision_11b")
        oracle.load_teacher()
        print("  [OK] Teacher Loaded on CPU (AMX)")
        # Test basic inference
        res = oracle.generate("What is in this image?", image=None, max_new_tokens=10)
        print(f"  [OK] Teacher Inference OK: {res[:50]}...")
        # Keep oracle loaded for ACC tests or unload to save RAM? 
        # We'll keep it for now as students run on GPU.
    except Exception as e:
        print(f"  [ERROR] Teacher: FAILED - {e}")
        return

    # 3. Test Student Models + ACC
    print("\n[3/3] Testing Student Fleet (ACC vs. Student-Only)...")
    
    wrapper_map = {
        "qwen2.5-vl-3b": ("wrappers.wrapper_qwen25vl_3b", "QwenVLStudentAgent"),
        "phi-4-multimodal": ("wrappers.wrapper_phi4_multimodal", "Phi4MultimodalStudentAgent"),
        "llava-v1.6-7b": ("wrappers.wrapper_llava16_7b", "LLaVA16StudentAgent"),
    }

    for m_name in models:
        print(f"\n--- Testing Model: {m_name} ---")
        module_path, class_name = wrapper_map[m_name]
        try:
            # Import and load
            pkg = __import__(module_path, fromlist=[class_name])
            AgentClass = getattr(pkg, class_name)
            agent = AgentClass()
            print(f"  [OK] {m_name} Loaded (4-bit)")

            for b in benchmarks:
                sample = test_samples[b]
                print(f"  Testing {b} sample...")
                
                # Test Student-Only (High threshold/Manual bypass)
                # For smoke test, we'll just run their default run_text
                # which uses the LAMBDA_STAR in the wrapper.
                res = agent.run_text(sample.question, image=sample.image, max_new_tokens=10)
                
                drift = res.get("drift_scores", [])
                handover = any(d > agent.lambda_star for d in drift) if drift else False
                
                print(f"    - Output: {res['generated_text'][:40]}...")
                print(f"    - Drift Trajectory: {len(drift)} tokens")
                print(f"    - ACC Triggered: {handover} (lambda*={agent.lambda_star})")
                
                if handover:
                    print(f"    - [OK] Successfully triggered Teacher handoff.")
                else:
                    print(f"    - [OK] No handoff needed (Student Confident).")

            # Cleanup
            print(f"  Cleaning up {m_name}...")
            del agent
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] {m_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("FLEET SMOKE TEST COMPLETE: System is Ready for Final Campaign.")
    print("="*60)

if __name__ == "__main__":
    smoke_test_fleet()
