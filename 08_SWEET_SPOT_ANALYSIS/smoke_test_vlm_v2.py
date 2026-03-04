import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def run_sub_test(model_name):
    print(f"\n--- Testing {model_name} in subprocess ---")
    script = f"""
import sys
from pathlib import Path
import torch
ROOT_DIR = Path('{ROOT_DIR}')
sys.path.insert(0, str(ROOT_DIR / '02_SRC'))

from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
from wrappers.wrapper_llava16_7b import LLaVA16StudentAgent

IMAGE_PATH = "/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/mathvista/images/1.jpg"

try:
    if '{model_name}' == 'qwen':
        agent = QwenVLStudentAgent()
    elif '{model_name}' == 'phi4':
        agent = Phi4MultimodalStudentAgent()
    elif '{model_name}' == 'llava':
        agent = LLaVA16StudentAgent()
    
    # Text-only
    print("[{model_name}] Running text-only...")
    res = agent.run_text("Describe this image.", image=None, max_new_tokens=5)
    print(f"[{model_name}] Text-only success: {{res['generated_text']}}")
    
    # Multimodal
    if Path(IMAGE_PATH).exists():
        print("[{model_name}] Running multimodal...")
        res = agent.run_text("What is in this image?", image=IMAGE_PATH, max_new_tokens=5)
        print(f"[{model_name}] Multimodal success: {{res['generated_text']}}")
    else:
        print(f"[{model_name}] Skipping multimodal (image missing)")
except Exception as e:
    print(f"[{model_name}] FAILED: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    with open("temp_test.py", "w") as f:
        f.write(script)
    
    res = subprocess.run([sys.executable, "temp_test.py"], capture_output=False)
    return res.returncode == 0

def main():
    models = ["qwen", "phi4", "llava"]
    for m in models:
        success = run_sub_test(m)
        if not success:
            print(f"!!! {m} test failed !!!")
            # sys.exit(1) # Continue to others to see full extent

if __name__ == "__main__":
    main()
