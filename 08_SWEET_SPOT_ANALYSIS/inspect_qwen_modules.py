import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import os

MODEL_PATH = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b"

print(f"Loading Qwen2.5-VL-3B to inspect modules...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    trust_remote_code=True
)

print("\n--- Visual Modules ---")
for name, module in model.named_modules():
    if "visual" in name and any(x in name for x in ["q_proj", "qkv", "attn", "blocks", "resblocks"]):
        print(f"Name: {name} | Type: {type(module)}")

print("\n--- End of Modules ---")
