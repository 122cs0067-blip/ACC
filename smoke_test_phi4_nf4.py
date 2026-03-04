#!/usr/bin/env python3
"""Quick NF4 loading smoke test for Phi-4 Multimodal."""
import os, sys, torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "02_SRC"))

MODEL_DIR = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal"

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["lm_head", "embed_tokens_extend"],
)

print("Loading Phi-4-MM in NF4...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    quantization_config=bnb_cfg,
    device_map="auto",
    attn_implementation="sdpa",
    output_hidden_states=True,
)

params = sum(p.numel() for p in model.parameters())
print(f"Successfully loaded in NF4!")
print(f"Total Parameters: {params:,} ({params/1e9:.2f}B)")
print(f"Model dtype: {model.dtype}")

# Check VRAM usage
if torch.cuda.is_available():
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM allocated: {vram:.2f} GB")
