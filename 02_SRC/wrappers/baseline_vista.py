import os
import sys
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from pathlib import Path

# Add project roots
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))
VISTA_PATH = "/home/cse-sdpl/research/ACC/03_BASELINES/VISTA"
sys.path.append(VISTA_PATH)

from transformers import AutoProcessor, AutoModelForVision2Seq
import gc

# VISTA Imports (from cloned repo)
try:
    from steering_vector import obtain_vsv, add_logits_flag, remove_logits_flag
    from llm_layers import add_vsv_layers, remove_vsv_layers
except ImportError:
    print("VISTA repository not found or path incorrect. Please ensure it's in 03_BASELINES/VISTA")

class VISTAAgent:
    """
    VISTA Baseline Wrapper (ICML 2025)
    'The Hidden Life of Tokens: Reducing Hallucination of LVLMs Via Visual Information Steering'
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        
        # VISTA Parameters
        self.vsv_lambda = 0.01      # Default from paper
        self.logits_aug = True
        self.logits_alpha = 0.3     # Default from paper
        self.tar_layers = "25,30"   # Default for 7B/32-layer models

    def load_model(self):
        if self.model is not None:
            return
            
        print(f"[VISTA] Loading {self.model_id}...")
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
        
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            quantization_config=bnb_cfg,
            device_map={"": self.device},
            trust_remote_code=True
        )
        self.model.eval()
        
        # Determine target layers for SLA based on model size
        num_layers = self._get_num_layers()
        if num_layers:
            self.tar_layers = f"{num_layers-7},{num_layers-2}"
            print(f"[VISTA] Model has {num_layers} layers. SLA target: {self.tar_layers}")

        self._patch_model_for_sla()

    def _get_num_layers(self) -> Optional[int]:
        try:
            from llm_layers import get_layers
            layers = get_layers(self.model)
            return len(layers)
        except:
            return None

    def _patch_model_for_sla(self):
        """Monkey-patch model for Self-Logits Augmentation (SLA)."""
        original_forward = self.model.forward
        
        def patched_forward(*args, **kwargs):
            if getattr(self.model, 'v_logits_aug', False):
                kwargs['output_hidden_states'] = True
            
            outputs = original_forward(*args, **kwargs)
            
            if getattr(self.model, 'v_logits_aug', False) and hasattr(outputs, 'hidden_states'):
                # SLA logic
                # hidden_states: tuple of (batch, seq, dim)
                h_states = outputs.hidden_states[1:] # ignore embedding
                lm_head = self._get_lm_head()
                
                if lm_head:
                    s_idx, e_idx = map(int, self.model.v_logits_layers.split(','))
                    tar_layers = list(range(max(0, s_idx), min(len(h_states), e_idx + 1)))
                    
                    if tar_layers:
                        # Compute intermediate logits
                        # [num_tar, batch, seq, vocab]
                        aug_logits = torch.stack([lm_head(h_states[idx]) for idx in tar_layers])
                        avg_aug_logits = aug_logits.mean(dim=0)
                        
                        alpha = self.model.v_logits_alpha
                        outputs.logits = alpha * avg_aug_logits + (1 - alpha) * outputs.logits
            
            return outputs

        self.model.forward = patched_forward

    def _get_lm_head(self):
        """Robustly find the LM head."""
        for name, module in self.model.named_modules():
            if name.endswith("lm_head"):
                return module
        return None

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()

    def run_inference(self, prompt: str, image=None, max_tokens: int = 256) -> Tuple[str, bool]:
        if self.model is None:
            self.load_model()
            
        # 1. Prepare Pos (Standard) and Neg (Text-Only) inputs
        # For LLaVA-1.6, AutoProcessor handles image placeholders
        inputs_pos = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        inputs_neg = self.processor(text=prompt, images=None, return_tensors="pt").to(self.device)
        
        # 2. Derive Visual Steering Vector (VSV)
        # Note: obtain_vsv uses model's forward. We must ensure it doesn't trigger SLA yet.
        self.model.v_logits_aug = False
        
        try:
            from steering_vector import obtain_vsv
            from llm_layers import add_vsv_layers, remove_vsv_layers
            
            kwargs_list = [[dict(inputs_neg), dict(inputs_pos)]]
            # We pass the whole model as obtain_vsv calls forward
            visual_vector, _ = obtain_vsv(None, self.model, kwargs_list)
            
            # 3. Apply Steering
            # VSV is injected into LLM layers
            llm_part = self.model
            if hasattr(self.model, 'language_model'):
                llm_part = self.model.language_model
                
            add_vsv_layers(llm_part, torch.stack([visual_vector], dim=1).to(self.device), [self.vsv_lambda])
            
            # 4. Enable SLA
            self.model.v_logits_aug = True
            self.model.v_logits_layers = self.tar_layers
            self.model.v_logits_alpha = self.logits_alpha
            
            # 5. Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs_pos,
                    max_new_tokens=max_tokens,
                    do_sample=False, # Greed/Beam as per user prefs
                    use_cache=True
                )
            
            # 6. Cleanup
            remove_vsv_layers(llm_part)
            self.model.v_logits_aug = False
            
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            return generated_text, False
            
        except Exception as e:
            print(f"[VISTA] Core execution failed: {e}. Falling back to standard.")
            with torch.no_grad():
                outputs = self.model.generate(**inputs_pos, max_new_tokens=max_tokens, do_sample=False)
            return self.processor.decode(outputs[0], skip_special_tokens=True), False

if __name__ == "__main__":
    # Smoke test
    agent = VISTAAgent("llava-hf/llava-v1.6-vicuna-7b-hf")
    # agent.load_model() # Optional as it lazy loads
    text, chasm = agent.run_inference("What is in this image?", image=None)
    print(f"Result: {text}")
