#!/usr/bin/env python3
import argparse
import os
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_activations(
    model_id: str,
    prompts: List[str],
    output_path: str,
    max_tokens: int = 50,
    use_awq: bool = False,
    device: str = "cuda",
    force_cpu: bool = False,
) -> None:
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    actual_device = "cpu" if force_cpu else device
    
    if use_awq:
        try:
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_quantized(
                model_id,
                fuse_layers=True,
                device_map=actual_device,
            )
        except ImportError:
            print("AWQ not available, falling back to AutoModelForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=actual_device,
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=actual_device,
            torch_dtype=torch.float16,
        )
    
    all_activations = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        inputs = tokenizer(prompt, return_tensors="pt").to(actual_device)
        input_ids = inputs.input_ids
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                
                hidden_state = outputs.hidden_states[-1][:, -1, :].float().cpu().numpy()
                all_activations.append(hidden_state.flatten())
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
    
    activations_array = np.array(all_activations, dtype=np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, activations_array)
    print(f"Saved {len(all_activations)} activation vectors to {output_path}")
    print(f"Shape: {activations_array.shape}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--use-awq", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    
    if args.prompts_file and os.path.isfile(args.prompts_file):
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "The scientific method is a systematic approach to",
            "In artificial intelligence, neural networks are",
            "Quantum mechanics describes the behavior of",
            "The theory of evolution explains how species",
            "Climate change is driven by increasing levels of",
        ]
    
    generate_activations(
        model_id=args.model_id,
        prompts=prompts,
        output_path=args.output,
        max_tokens=args.max_tokens,
        use_awq=args.use_awq,
        device=args.device,
        force_cpu=args.force_cpu,
    )

if __name__ == "__main__":
    main()
