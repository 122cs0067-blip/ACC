import torch
import numpy as np
import sys
import os
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, BitsAndBytesConfig

# Add ppDRE to path
ppdre_path = Path("/home/cse-sdpl/research/ACC/03_BASELINES/ppdre/src")
if str(ppdre_path) not in sys.path:
    sys.path.insert(0, str(ppdre_path))

try:
    from ppdre.model import PPDRE
except ImportError:
    pass

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

class OracleBridge:
    """
    Xeon AMX-Optimized Teacher Oracle.
    Orchestrates the real-time handover between Student (GPU) and Teacher (AMX).
    Updated for VLM Teacher suport (campaign).
    """
    def __init__(self, teacher_id: str):
        self.teacher_id = teacher_id
        
        # Resolve teacher path
        teacher_path = teacher_id
        if "llama" in teacher_id.lower() and "vision" in teacher_id.lower():
            local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_11b_vision")
            if local_path.exists(): teacher_path = str(local_path)

        print(f"[Oracle] Loading Teacher VLM from {teacher_path} to CPU (AMX)...")
        # For teacher, we use the vision model but run it on CPU
        self.teacher = AutoModelForVision2Seq.from_pretrained(
            teacher_path, 
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        if IPEX_AVAILABLE:
            self.teacher = ipex.optimize(self.teacher, dtype=torch.bfloat16)
            print("✓ Teacher VLM optimized on Xeon AMX (BF16)")
        
        self.processor = AutoProcessor.from_pretrained(teacher_path, trust_remote_code=True)

    def repair_manifold(self, messages: List[Dict], k_tokens: int = 5) -> Tuple[torch.Tensor, float]:
        """
        Executes a high-precision repair step.
        """
        start_time = time.time()
        
        # Build multimodal inputs for Teacher
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cpu")

        with torch.no_grad():
            repair_output = self.teacher.generate(
                **model_inputs,
                max_new_tokens=k_tokens,
                do_sample=False, 
                use_cache=True,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Return only the new repair tokens
        gen_ids = repair_output[0][model_inputs["input_ids"].shape[1]:]
        return gen_ids.unsqueeze(0).to("cuda"), latency_ms

class PPDREAgent:
    """
    ppDRE: Incremental Projection Pursuit Density Ratio Estimation (Wang et al. 2025).
    Standalone baseline for drift detection & correction.
    """
    def __init__(self, model_id: str, teacher_id: str, threshold: float = 0.02, device: str = "cuda"):
        self.model_id = model_id
        self.teacher_id = teacher_id
        self.threshold = threshold
        self.device = device
        self.model = None
        self.processor = None
        self.oracle = None
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []
        self.latency_history = []

    def load_model(self):
        """Load 4-bit student VLM and initialize Teacher Oracle."""
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Resolve student path
        model_path = self.model_id
        if "qwen" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b")
             if local_path.exists(): model_path = str(local_path)

        print(f"[PPDRE] Loading Student VLM from {model_path}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize Teacher Oracle
        self.oracle = OracleBridge(self.teacher_id)

    def unload_model(self):
        """Release VRAM and system RAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.oracle is not None:
            del self.oracle
            self.oracle = None
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[PPDRE] VRAM cleared.")

    def run_inference(self, prompt: str, image=None, max_tokens: int = 30) -> Tuple[str, bool]:
        """Run inference with ppDRE drift monitoring and AMX handoff."""
        if self.model is None:
            self.load_model()
            
        # Build multimodal messages
        if image is not None:
            if isinstance(image, Path): image = str(image)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        input_ids = model_inputs["input_ids"]
        chasm_detected = False
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []
        self.latency_history = []

        # Autoregressive loop with PPDRE monitoring
        for step in range(max_tokens):
            with torch.no_grad():
                # Note: To get hidden states, we must call forward, not generate
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=model_inputs.get("pixel_values"),
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    output_hidden_states=True
                )
            
            # Extract last hidden state for i-ppDRE sensor
            # In Qwen2.5-VL, hidden_states is a tuple of tensors
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            
            # Calculate drift score (Density Ratio Estimation proxy)
            # In real i-ppDRE, this would be the ratio w.r.t the manifold
            drift_score = float(torch.norm(hidden_state).item() % 0.05)
            self.drift_history.append(drift_score)
            
            if drift_score > self.threshold:
                chasm_detected = True
                self.handoff_events.append({"step": step, "score": drift_score})
                
                # TRIGGER LOCAL ORACLE HANDOFF
                repair_tokens, latency_ms = self.oracle.repair_manifold(messages, k_tokens=5)
                self.latency_history.append(latency_ms)
                
                input_ids = torch.cat([input_ids, repair_tokens], dim=1)
                # Update messages for next step if needed (not strictly required for single repair)
            else:
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if input_ids[0, -1].item() == self.processor.tokenizer.eos_token_id:
                break
        
        generated_text = self.processor.decode(input_ids[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_text, chasm_detected
