import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal"

def test_load_full():
    print(f"Attempting to load Phi-4 with full configuration from {MODEL_ID}")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["embed_tokens_extend"],
    )

    try:
        print("Starting from_pretrained...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            output_hidden_states=True,
        )
        print("Model loaded.")
        
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Processor loaded.")

        # Hook registration logic from wrapper
        captured_vit_q = None
        def hook_fn(module, input, output):
            nonlocal captured_vit_q
            captured_vit_q = output.detach().float().cpu()

        print("Registering hooks...")
        hook_registered = False
        target = None
        for name, module in model.named_modules():
            if ("image_embed" in name or "visual" in name) and "q_proj" in name:
                target = module 
        
        if target is not None:
            target.register_forward_hook(hook_fn)
            hook_registered = True
            print("[Phi4-MM] Registered hook on vision module.")
        
        print("Initialization complete.")

        # Test a small generation to trigger hooks
        print("Testing generation...")
        inputs = processor(text="<|image_1|>\nWhat is in this image?", images=None, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=5, output_hidden_states=True, return_dict_in_generate=True)
        print("Generation successful.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_full()
