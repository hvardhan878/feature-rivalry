import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DTYPE


def load_model_and_tokenizer(model_name: str):
    """Load Gemma-2-2B-IT model and tokenizer onto CUDA in bfloat16."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    print(f"Loading model: {model_name} (bfloat16, this may take a minute...)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DTYPE,
        trust_remote_code=True,
    )
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded on CUDA.")
    return model, tokenizer
