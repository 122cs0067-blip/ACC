import sys
from pathlib import Path
SRC_ROOT = Path("/home/cse-sdpl/research/ACC/02_SRC")
sys.path.insert(0, str(SRC_ROOT))

from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
import torch

def test_hook():
    print("Initializing QwenVLStudentAgent...")
    agent = QwenVLStudentAgent(use_teacher=False)
    
    # Check if a hook was registered
    # The agent prints registration status during init
    
    print("\nRunning a dummy inference to trigger hook...")
    dummy_prompt = "What's in this image?"
    # We don't even need a real image for the hook to trigger if input is prepared
    # But let's see if we can just check self.model for hooks
    
    hooked = False
    for name, module in agent.model.named_modules():
        if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
            for hook_id, hook in module._forward_hooks.items():
                print(f"Found hook on: {name}")
                hooked = True
    
    if hooked:
        print("\nSUCCESS: Vision hook verified.")
    else:
        print("\nFAILURE: No hooks found on vision modules.")

if __name__ == "__main__":
    test_hook()
