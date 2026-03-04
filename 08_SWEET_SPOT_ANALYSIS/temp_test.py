
import sys
from pathlib import Path
import torch
ROOT_DIR = Path('/home/cse-sdpl/research/ACC')
sys.path.insert(0, str(ROOT_DIR / '02_SRC'))

from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
from wrappers.wrapper_llava16_7b import LLaVA16StudentAgent

IMAGE_PATH = "/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/mathvista/images/1.jpg"

try:
    if 'llava' == 'qwen':
        agent = QwenVLStudentAgent()
    elif 'llava' == 'phi4':
        agent = Phi4MultimodalStudentAgent()
    elif 'llava' == 'llava':
        agent = LLaVA16StudentAgent()
    
    # Text-only
    print("[llava] Running text-only...")
    res = agent.run_text("Describe this image.", image=None, max_new_tokens=5)
    print(f"[llava] Text-only success: {res['generated_text']}")
    
    # Multimodal
    if Path(IMAGE_PATH).exists():
        print("[llava] Running multimodal...")
        res = agent.run_text("What is in this image?", image=IMAGE_PATH, max_new_tokens=5)
        print(f"[llava] Multimodal success: {res['generated_text']}")
    else:
        print(f"[llava] Skipping multimodal (image missing)")
except Exception as e:
    print(f"[llava] FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
