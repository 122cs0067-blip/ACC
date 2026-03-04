"""
Quantize ALL student VLMs to AWQ 4-bit (GEMM, group_size=128).

Models:
  1. Phi-4 Multimodal (phi4mm -> phi3_v backend)
  2. LLaVA-v1.6-7B (llava_next, native support)
  3. Qwen2.5-VL-3B (qwen2_5_vl, native support)

This creates pre-quantized checkpoints for the final campaign.
"""
import sys
import os
import shutil
import torch
from pathlib import Path

BASE = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm")

MODELS = {
    "phi4_multimodal": {
        "src": BASE / "phi4_multimodal",
        "dst": BASE / "phi4_multimodal_awq",
        "needs_patch": True,
        "trust_remote_code": True,
        "extra_files": [
            "preprocessor_config.json", "processor_config.json",
            "configuration_phi4mm.py", "modeling_phi4mm.py",
            "processing_phi4mm.py", "vision_siglip_navit.py",
            "speech_conformer_encoder.py", "special_tokens_map.json",
            "tokenizer_config.json", "merges.txt", "vocab.json",
            "added_tokens.json",
        ],
        "extra_dirs": ["vision-lora", "speech-lora"],
    },
    "llava16_7b": {
        "src": BASE / "llava16_7b",
        "dst": BASE / "llava16_7b_awq",
        "needs_patch": False,
        "trust_remote_code": False,
        "extra_files": [
            "preprocessor_config.json", "processor_config.json",
            "special_tokens_map.json", "tokenizer_config.json",
            "chat_template.json",
        ],
        "extra_dirs": [],
    },
    "qwen25vl_3b": {
        "src": BASE / "qwen25vl_3b",
        "dst": BASE / "qwen25vl_3b_awq",
        "needs_patch": False,
        "trust_remote_code": True,
        "extra_files": [
            "preprocessor_config.json", "chat_template.json",
            "tokenizer_config.json", "merges.txt", "vocab.json",
        ],
        "extra_dirs": [],
    },
}

QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

CALIB_DATA = [
    "The image shows a cat sitting on a mat.",
    "Is there a person in this photograph?",
    "Describe the objects visible in the scene.",
    "What color is the sky in this image?",
    "The capital of France is Paris.",
    "Calculate the sum of 3 and 7.",
    "A dog is running through a green field.",
    "The weather today is sunny with clear skies.",
]


def quantize_model(name, cfg):
    print(f"\n{'='*60}")
    print(f"Quantizing: {name}")
    print(f"{'='*60}")

    if cfg["dst"].exists():
        print(f"  [SKIP] AWQ checkpoint already exists at {cfg['dst']}")
        return True

    # Register phi4mm if needed
    if cfg["needs_patch"]:
        from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
        from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
        AWQ_CAUSAL_LM_MODEL_MAP["phi4mm"] = AWQ_CAUSAL_LM_MODEL_MAP["phi3_v"]
        TRANSFORMERS_AUTO_MAPPING_DICT["phi4mm"] = "AutoModelForCausalLM"
        print("  Registered phi4mm -> phi3_v in AutoAWQ")

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, AutoProcessor

    print(f"  Loading FP16 model from {cfg['src']}...")
    model = AutoAWQForCausalLM.from_pretrained(
        str(cfg["src"]),
        trust_remote_code=cfg["trust_remote_code"],
        safetensors=True,
    )

    # Get tokenizer
    try:
        processor = AutoProcessor.from_pretrained(
            str(cfg["src"]), trust_remote_code=cfg["trust_remote_code"]
        )
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            str(cfg["src"]), trust_remote_code=cfg["trust_remote_code"]
        )

    print(f"  Quantizing (AWQ w4, GEMM, group_size=128)...")
    model.quantize(tokenizer, quant_config=QUANT_CONFIG, calib_data=CALIB_DATA)

    print(f"  Saving to {cfg['dst']}...")
    cfg["dst"].mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(cfg["dst"]))
    tokenizer.save_pretrained(str(cfg["dst"]))

    # Copy extra files
    for f in cfg.get("extra_files", []):
        src = cfg["src"] / f
        if src.exists():
            shutil.copy2(str(src), str(cfg["dst"] / f))
    for d in cfg.get("extra_dirs", []):
        src = cfg["src"] / d
        if src.exists():
            dst = cfg["dst"] / d
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))

    print(f"  DONE: {name}")
    return True


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        for name, cfg in MODELS.items():
            try:
                quantize_model(name, cfg)
            except Exception as e:
                print(f"  FAILED: {name} -> {e}")
    elif target in MODELS:
        quantize_model(target, MODELS[target])
    else:
        print(f"Unknown model: {target}. Choose from: {list(MODELS.keys())} or 'all'")


if __name__ == "__main__":
    main()
