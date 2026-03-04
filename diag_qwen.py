from transformers import Qwen2VLForConditionalGeneration
import torch

model_path = '/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map='auto',
    load_in_4bit=True
)

print("-" * 40)
print("Qwen2.5-VL Vision Model named modules:")
for name, module in model.named_modules():
    if 'visual' in name and 'q_proj' in name:
        print(f"Name: {name} | Type: {type(module)}")

print("-" * 40)
print("Checking last block path:")
try:
    print(f"Blocks type: {type(model.visual.blocks)}")
    print(f"Last block type: {type(model.visual.blocks[-1])}")
    print(f"Attn type: {type(model.visual.blocks[-1].attn)}")
    print(f"QProj type: {type(model.visual.blocks[-1].attn.q_proj)}")
except Exception as e:
    print(f"Error accessing path: {e}")
