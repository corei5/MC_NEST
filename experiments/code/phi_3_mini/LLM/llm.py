import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)

"""Set up environment configurations."""
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.random.manual_seed(0)
set_seed(42)    


def initialize_quantization():
    """Initialize BitsAndBytesConfig for quantization (if needed)."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return quantization_config

def load_llm_Phi_3_mini():
    MODEL_PATH = "microsoft/Phi-3-mini-128k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=True,
        
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype='auto',
        device_map=device_map,
        max_new_tokens=4000
    )

    return tokenizer, model, pipeline, MODEL_PATH




