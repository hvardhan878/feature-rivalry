"""
Sanity check — verifies every component of the Feature Rivalry pipeline.
Should complete in under 2 minutes on M1 Pro.
Run: python sanity_check.py
"""

import sys
import os

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step(n, description):
    print(f"\n[Step {n}] {description}")


def fail(n, e):
    print(f"FAIL at step {n}: {e}")
    sys.exit(1)


# ─── Step 1: Torch + MPS ────────────────────────────────────────────────────
step(1, "Torch + MPS check")
try:
    import torch
    assert torch.backends.mps.is_available(), "MPS not available"
    x = torch.tensor([1.0, 2.0]).to("mps")
    assert x.device.type == "mps"
    print("PASS: MPS available and working")
except Exception as e:
    fail(1, e)

# ─── Step 2: Model load ──────────────────────────────────────────────────────
step(2, "Model load check")
try:
    from model.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer("google/gemma-2-2b-it")
    assert model is not None
    first_param_device = next(model.parameters()).device.type
    assert first_param_device == "mps", f"Model on wrong device: {first_param_device}"
    print("PASS: Model loaded on MPS")
except Exception as e:
    fail(2, e)

# ─── Step 3: Single forward pass ────────────────────────────────────────────
step(3, "Single forward pass check")
try:
    prompt = "Q: What is the capital of France?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    attention_mask = inputs["attention_mask"].to("mps")
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask,
                                max_new_tokens=5, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    assert "Paris" in decoded or len(decoded) > len(prompt)
    print(f"PASS: Forward pass works. Output: {decoded}")
except Exception as e:
    fail(3, e)

# ─── Step 4: ActivationCache ─────────────────────────────────────────────────
step(4, "ActivationCache check")
try:
    from model.hooks import ActivationCache
    test_layers = [0, 10, 24]
    inputs = tokenizer("Q: What is the capital of France?\nA:", return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    attention_mask = inputs["attention_mask"].to("mps")
    with ActivationCache(model, test_layers) as cache:
        with torch.no_grad():
            model.generate(input_ids, attention_mask=attention_mask,
                           max_new_tokens=1, do_sample=False)
        for layer in test_layers:
            hidden = cache.get_activations(layer)
            assert hidden is not None, f"No activation for layer {layer}"
            assert hidden.shape == (2304,), f"Wrong shape: {hidden.shape}"
    print("PASS: ActivationCache works, hidden dim = 2304")
except Exception as e:
    fail(4, e)

# ─── Step 5: SAE load + feature extraction ───────────────────────────────────
step(5, "SAE load + feature extraction check")
try:
    import numpy as np
    from sae.gemma_scope import load_sae, get_feature_activations
    sae = load_sae(
        layer_idx=10,
        release="gemma-scope-2b-pt-res",
        sae_id="layer_10/width_16k/average_l0_77",
    )
    assert sae is not None
    inputs = tokenizer("Q: What is the capital of France?\nA:", return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    attention_mask = inputs["attention_mask"].to("mps")
    with ActivationCache(model, [10]) as cache:
        with torch.no_grad():
            model.generate(input_ids, attention_mask=attention_mask,
                           max_new_tokens=1, do_sample=False)
        hidden = cache.get_activations(10)
    features = get_feature_activations(sae, hidden)
    assert features.shape == (16384,), f"Wrong feature shape: {features.shape}"
    n_active = int(np.sum(features > 0))
    print(f"PASS: SAE works. {n_active} features active out of 16384")
except Exception as e:
    fail(5, e)

# ─── Step 6: Dataset load ────────────────────────────────────────────────────
step(6, "Dataset load check")
try:
    from data.dataset_builder import load_unambiguous_sample, load_pop_qa_sample
    unambiguous = load_unambiguous_sample(5)
    popqa = load_pop_qa_sample(5)
    assert len(unambiguous) == 5
    assert len(popqa) == 5
    assert "question" in unambiguous[0] and "answer" in unambiguous[0]
    assert "question" in popqa[0] and "answer" in popqa[0]
    print("PASS: Datasets load.")
    print(f"      Example unambiguous: {unambiguous[0]['question']}")
    print(f"      Example PopQA:    {popqa[0]['question']}")
except Exception as e:
    fail(6, e)

# ─── Step 7: Entropy measurement ─────────────────────────────────────────────
step(7, "Entropy measurement check")
try:
    from data.dataset_builder import measure_answer_entropy
    entropy = measure_answer_entropy(
        model,
        tokenizer,
        question="What is 2 + 2?",
        n_samples=3,
        temperature=1.0,
        max_new_tokens=5,
    )
    assert isinstance(entropy, float)
    assert entropy >= 0.0
    print(f"PASS: Entropy measurement works. Entropy for '2+2': {entropy:.3f} (expect ~0)")
except Exception as e:
    fail(7, e)

# ─── Step 8: Rivalry score (synthetic data) ───────────────────────────────────
step(8, "Rivalry score check (synthetic data)")
try:
    import numpy as np
    from analysis.rivalry_score import compute_rivalry_score, find_top_rival_pairs

    # Inject a known rival pair: features 0 and 1 are strongly anti-correlated
    fake_matrix = np.random.randn(20, 50)
    fake_matrix[:, 1] = -fake_matrix[:, 0] + np.random.randn(20) * 0.1

    corr = np.corrcoef(fake_matrix.T)
    upper = corr[np.triu_indices_from(corr, k=1)]

    score = compute_rivalry_score(upper)
    assert score < 0, f"Rivalry score should be negative, got {score}"

    pairs = find_top_rival_pairs(corr, np.arange(50), top_k=5)
    assert len(pairs) == 5
    assert pairs[0]["correlation"] < pairs[-1]["correlation"]

    top_pair_indices = {pairs[0]["feature_i"], pairs[0]["feature_j"]}
    assert top_pair_indices == {0, 1}, (
        f"Expected injected rival pair at top, got {top_pair_indices}"
    )
    print(
        f"PASS: Rivalry score = {score:.3f}, "
        "injected rival pair correctly identified as top pair"
    )
except Exception as e:
    fail(8, e)

# ─── Step 9: Results directory ───────────────────────────────────────────────
step(9, "Results directory check")
try:
    os.makedirs("results/figures", exist_ok=True)
    assert os.path.exists("results/figures")
    print("PASS: Results directory ready")
except Exception as e:
    fail(9, e)

# ─── Final summary ────────────────────────────────────────────────────────────
print()
print("=" * 40)
print("ALL CHECKS PASSED")
print("Setup is ready. Run next:")
print("  python run.py --debug --exp 1")
print("=" * 40)
