
import sys
from pathlib import Path
import torch

print("DEBUG: Script started")

PROJECT_ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.insert(0, str(PROJECT_ROOT / "02_SRC"))

def test_model(module_name, agent_cls_name):
    print(f"\n--- Testing {module_name} ---")
    try:
        full_module_name = f"wrappers.{module_name}"
        
        # Use importlib for cleaner imports
        import importlib
        mod = importlib.import_module(full_module_name)
        agent_cls = getattr(mod, agent_cls_name)
        
        print(f"DEBUG: Initializing {agent_cls_name}...")
        agent = agent_cls(use_teacher=False)
        print(f"DEBUG: Success for {module_name}!")
        agent.unload_model()
    except Exception as e:
        import traceback
        print(f"DEBUG: Failed for {module_name}: {e}")
        traceback.print_exc()

test_model("wrapper_phi4_multimodal", "Phi4MultimodalStudentAgent")
test_model("wrapper_llava16_7b", "LLaVA16StudentAgent")

print("\nDEBUG: Script finished")
