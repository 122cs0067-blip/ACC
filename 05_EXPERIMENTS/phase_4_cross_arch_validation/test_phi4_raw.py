
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
from pathlib import Path

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
LOCAL_DIR = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal")
model_path = str(LOCAL_DIR) if LOCAL_DIR.exists() else MODEL_ID

print(f"DEBUG: Loading raw model from {model_path}...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

prompt = "Is there a backpack in the image?"
image_path = Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/pope/images/2619.jpg")
image = Image.open(image_path).convert("RGB")

messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
text_input = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(text=text_input, images=image, return_tensors="pt").to("cuda")

print("DEBUG: Generating...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

prediction = processor.tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"PREDICTION: {prediction}")
