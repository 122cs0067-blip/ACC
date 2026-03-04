import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os

# Use the user's snippet
model_id = "microsoft/Phi-4-multimodal-instruct"
local_dir = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal"

path = local_dir if os.path.exists(local_dir) else model_id

print(f"Loading model from: {path}")

try:
    # Load config first to see if there is any quantization info
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    print(f"Config quantization_config: {getattr(config, 'quantization_config', 'None')}")
    
    # Load model with user's specific params
    # Note: we use device_map="cpu" to avoid VRAM issues during this check
    # and trust_remote_code=True as requested.
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        trust_remote_code=True, 
        dtype="auto", 
        device_map="cpu"
    )
    
    print(f"Model class: {type(model)}")
    print(f"Model dtype: {model.dtype}")
    
    # Check for bitsandbytes (NF4) or autoawq (AWQ) attributes
    is_bnb = hasattr(model, "is_quantized") and model.is_quantized
    is_awq = "AWQ" in str(type(model)) or hasattr(model, "quant_config")
    
    print(f"Is BitsAndBytes (NF4/Int8) quantized: {is_bnb}")
    print(f"Is AWQ quantized: {is_awq}")
    
    # Check layers
    for name, module in model.named_modules():
        if "q_proj" in name or "query_key_value" in name:
            print(f"Sample layer ({name}) type: {type(module)}")
            break

except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
