import torch
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
import os

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_DIR = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_path = LOCAL_DIR if os.path.exists(LOCAL_DIR) else MODEL_ID
print(f"Loading model from {model_path}...")

try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded successfully.")
    
    print("\nListing modules with 'visual' and 'q_proj':")
    for name, module in model.named_modules():
        if "visual" in name and "q_proj" in name:
            print(f"Name: {name} | Type: {type(module)}")

except Exception as e:
    print(f"Failed to load/inspect model: {e}")
    import traceback
    traceback.print_exc()
