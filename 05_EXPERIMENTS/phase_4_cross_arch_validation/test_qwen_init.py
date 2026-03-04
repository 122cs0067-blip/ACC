
import sys
from pathlib import Path

print("DEBUG: Script started")

# Add SRC to path
PROJECT_ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.insert(0, str(PROJECT_ROOT / "02_SRC"))
print(f"DEBUG: sys.path[0] = {sys.path[0]}")

try:
    print("DEBUG: Attempting import...")
    from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
    print("DEBUG: Import successful")
    
    print("Attempting to initialize QwenVLStudentAgent with anchoring...")
    agent = QwenVLStudentAgent(use_teacher=False)
    print("Success!")
    agent.unload_model()
except Exception as e:
    import traceback
    print(f"DEBUG: Caught exception: {e}")
    traceback.print_exc()
print("DEBUG: Script finished")
