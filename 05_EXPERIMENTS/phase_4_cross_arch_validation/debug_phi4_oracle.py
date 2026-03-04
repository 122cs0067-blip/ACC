
import sys
from pathlib import Path
from PIL import Image
import torch

PROJECT_ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.insert(0, str(PROJECT_ROOT / "02_SRC"))

from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent

def debug_phi4():
    print("DEBUG: Initializing Phi-4 Student...")
    # use_teacher=True will load the OracleBridge
    agent = Phi4MultimodalStudentAgent(use_teacher=True)
    
    prompt = "Is there a backpack in the image?"
    # Corrected path based on find_by_name
    image_path = PROJECT_ROOT / "01_DATA/Benchmarks/pope/images/2619.jpg"
    image = Image.open(image_path).convert("RGB")
    
    print(f"DEBUG: Running inference for prompt: {prompt}")
    
    # Run the agent
    result = agent.run_text(prompt, image=image, max_new_tokens=20, task_id="debug_phi4")
    
    print("\n--- RESULTS ---")
    print(f"Generated Text: {result['generated_text']}")
    print(f"ACC Active: {result['dcdr']}")
    print(f"Efficiency: {result['efficiency']}")
    print(f"Interventions: {result['interventions']}")
    
    agent.unload_model()

if __name__ == "__main__":
    debug_phi4()
