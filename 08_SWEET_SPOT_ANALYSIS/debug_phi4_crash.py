import sys
import os
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_SRC"))

from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent

def debug_phi4():
    print("Initializing Phi-4 Student Agent...")
    try:
        print("[DEBUG] Step 1: Loading model and processor...")
        agent = Phi4MultimodalStudentAgent(use_teacher=True, device="cpu") # Force CPU to avoid GPU issues during debug
        print("[DEBUG] Step 2: Model Loaded.")
        
        prompt = "Is there a backpack in the image?"
        print(f"[DEBUG] Step 3: Running inference on prompt: {prompt}")
        res = agent.run_text(prompt=prompt)
        print(f"[DEBUG] Step 4: Result: {res['generated_text']}")
        print(f"ACC Active: {res['acc_active']}")
        
    except Exception as e:
        import traceback
        print("\n[CRITICAL ERROR] Phi-4 Inference Failed:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_phi4()
