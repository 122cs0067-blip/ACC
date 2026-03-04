import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal"

def test_load():
    print(f"Attempting to load Phi-4 from {MODEL_ID}")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="gpu",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        print("Model loaded successfully.")
        
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Processor loaded successfully.")
    except Exception as e:
        print(f"Error during loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
