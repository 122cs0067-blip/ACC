#!/usr/bin/env python3
import os
import sys
import torch
import gc
from pathlib import Path

# Add Project Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "02_SRC"))

def smoke_test_agent(name, class_ref):
    print(f"\n" + "="*60)
    print(f"SMOKE TEST: {name}")
    print("="*60)
    try:
        # 1. Initialize
        print(f"[{name}] Loading model...")
        agent = class_ref(use_teacher=False)
        print(f"[{name}] Load successful.")
        
        # 2. Dummy Inference
        print(f"[{name}] Running dummy inference...")
        res = agent.run_text("Is this a test?", max_new_tokens=10, task_id="smoke_test")
        print(f"[{name}] Result: {res['generated_text']}")
        print(f"[{name}] Drift Scores Count: {len(res['drift_scores'])}")
        
        # 3. Unload
        print(f"[{name}] Unloading...")
        if hasattr(agent, 'unload_model'):
            agent.unload_model()
        del agent
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{name}] Integrity OK.")
        
    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
    from wrappers.wrapper_llava16_7b import LLaVA16StudentAgent
    
    # We don't test Qwen here because it's currently running in the background sweep
    smoke_test_agent("Phi-4-Multimodal", Phi4MultimodalStudentAgent)
    smoke_test_agent("LLaVA-v1.6-7B", LLaVA16StudentAgent)
    
    print("\n" + "="*60)
    print("FLEET INTEGRITY CHECK COMPLETE")
    print("="*60)
