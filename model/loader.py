import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    """Load Gemma-2-2B-IT model and tokenizer onto MPS device in float32."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    print(f"Loading model: {model_name} (float32, this may take a minute...)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to("mps")
    model.eval()
    print("Model loaded on MPS.")
    return model, tokenizer
