import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import os

MODEL_PATH = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b"

print(f"Loading Qwen2.5-VL-3B to inspect blocks.31...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    trust_remote_code=True
)

print("\n--- Blocks.31 Children ---")
target_block = None
if hasattr(model, 'visual') and hasattr(model.visual, 'blocks'):
    target_block = model.visual.blocks[-1]
elif hasattr(model.model, 'visual') and hasattr(model.model.visual, 'blocks'):
    target_block = model.model.visual.blocks[-1]

if target_block:
    for name, module in target_block.named_modules():
        print(f"Name: {name} | Type: {type(module)}")
else:
    print("Could not find visual blocks.")

print("\n--- End of Modules ---")
