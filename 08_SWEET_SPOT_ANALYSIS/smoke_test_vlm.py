import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "02_SRC"))

from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
from wrappers.wrapper_llava16_7b import LLaVA16StudentAgent

# Dummy image path
IMAGE_PATH = "/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/mathvista/images/1.jpg"

def smoke_test():
    models = ["qwen", "phi4", "llava"]
    for m in models:
        print(f"\n--- Testing {m} ---")
        try:
            if m == "qwen":
                agent = QwenVLStudentAgent()
            elif m == "phi4":
                agent = Phi4MultimodalStudentAgent()
            elif m == "llava":
                agent = LLaVA16StudentAgent()
            
            res = agent.run_text("Describe this image.", image=None, max_new_tokens=5)
            print(f"[{m}] Text-only success: {res['generated_text']}")
            
            # If image exists, test multimodal
            if Path(IMAGE_PATH).exists():
                res = agent.run_text("What is in this image?", image=IMAGE_PATH, max_new_tokens=5)
                print(f"[{m}] Multimodal success: {res['generated_text']}")
            else:
                print(f"[{m}] Skipping multimodal (image not found at {IMAGE_PATH})")
            
            agent.unload_model()
            del agent
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{m}] FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    smoke_test()
