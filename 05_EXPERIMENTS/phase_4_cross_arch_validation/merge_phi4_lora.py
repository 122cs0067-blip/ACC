"""
Merge Phi-4 Multimodal's vision LoRA into base weights and save for NF4-compatible loading.

Strategy:
  1. Load FP16 on CPU
  2. Activate vision LoRA
  3. Call merge() on each LoRA layer to fold adapter weights into base Linear
  4. Save the full model state dict + config (weights are now merged)
  5. Copy all processor/tokenizer/custom files from original directory

The resulting checkpoint has LoRA weights folded into the base layers.
When loaded, the LoRA layers will be initialized with zero deltas (identity),
so set_lora_adapter() becomes a no-op in practice.
"""
import sys
import torch
import shutil
import json
from pathlib import Path

MODEL_DIR = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal")
OUTPUT_DIR = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal_merged")


def merge():
    from transformers import AutoModelForCausalLM

    print(f"Step 1: Loading FP16 model on CPU from {MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Activate the vision adapter
    print("Step 2: Activating vision LoRA and merging...")
    model.set_lora_adapter("vision")

    from peft.tuners.lora.layer import LoraLayer
    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            try:
                module.merge()
                merged_count += 1
            except Exception as e:
                print(f"  Warning: Could not merge {name}: {e}")

    print(f"  Merged {merged_count} LoRA layers into base weights.")

    # Save the model (with merged weights)
    print(f"Step 3: Saving merged model to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)

    # Copy ALL files from original except .safetensors and .cache
    print("Step 4: Copying processor and custom files...")
    skip_extensions = {".safetensors"}
    skip_names = {".cache", ".git", ".gitattributes"}

    for item in MODEL_DIR.iterdir():
        if item.name in skip_names:
            continue
        if item.suffix in skip_extensions:
            continue
        dst = OUTPUT_DIR / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(item), str(dst))
            print(f"  Copied dir: {item.name}/")
        elif item.is_file() and not dst.exists():
            shutil.copy2(str(item), str(dst))
            print(f"  Copied: {item.name}")

    # Delete the large original .safetensors index if our save created a new one
    old_index = OUTPUT_DIR / "model.safetensors.index.json"
    if old_index.exists():
        print(f"  Model index exists at {old_index}")

    print(f"\nDone. Merged checkpoint at: {OUTPUT_DIR}")
    print("Load with NF4 quantization (no LoRA conflict).")

    # Free memory
    del model
    import gc; gc.collect()


if __name__ == "__main__":
    merge()
