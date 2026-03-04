import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os
import sys
from pathlib import Path

# Use local path
local_dir = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal"

print(f"Testing local load from: {local_dir}")

try:
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Load model with user's specific params
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, 
        trust_remote_code=True, 
        dtype="auto", 
        device_map="auto"
    )
    
    print(f"Successfully loaded {local_dir}")
    print(f"Model class: {type(model)}")
    print(f"Model dtype: {model.dtype}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
