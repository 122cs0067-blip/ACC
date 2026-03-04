import torch
import os
import time
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from typing import Tuple, List, Optional, Dict, Any
import sys
from pathlib import Path

# Add project root to sys.path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class Any4Agent:
    """
    Any4: Aggressive 4-bit quantization (Passive Baseline).
    This baseline uses standard NF4 quantization and acts as the 'no safety' control.
    Updated for VLM support (campaign).
    """
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []

    def load_model(self):
        """Load 4-bit VLM model into memory."""
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Check if model_id is a short name or path
        model_path = self.model_id
        if "qwen" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b")
             if local_path.exists(): model_path = str(local_path)
        elif "llava" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/llava16_7b")
             if local_path.exists(): model_path = str(local_path)
        elif "phi" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_mm")
             if local_path.exists(): model_path = str(local_path)

        print(f"[Any4] Loading VLM from {model_path}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map={"": self.device},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def unload_model(self):
        """Release VRAM and system RAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Any4] VRAM cleared.")

    def run_inference(self, prompt: str, image=None, max_tokens: int = 256) -> Tuple[str, bool]:
        """Run pure 4-bit inference without any interventions."""
        if self.model is None:
            self.load_model()
        
        # Build multimodal inputs
        if image is not None:
            if isinstance(image, Path):
                image = str(image)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Special handling for Qwen2.5-VL if needed, but AutoProcessor should handle it
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            model_inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            # Text-only fallback
            model_inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.device)

        input_ids = model_inputs["input_ids"]
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
            )
        
        # Decode only the NEW tokens
        gen_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.processor.decode(gen_ids, skip_special_tokens=True)
        return generated_text, False # No chasm detection in Any4
