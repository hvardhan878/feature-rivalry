import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from config import LAYERS_TO_ANALYZE, SAE_IDS, SAE_RELEASE
from model.hooks import ActivationCache
from sae.gemma_scope import get_feature_activations, load_sae


def _per_prompt_rivalry_score(sae, hidden_last_token: np.ndarray, top_k: int = 50) -> float:
    """
    Compute per-prompt rivalry score from last token hidden state.
    Takes top-k most active SAE features and computes 5th percentile
    of their pairwise correlations — mirrors Exp 1 population measure
    but applied per-prompt.
    Lower (more negative) = more rivalry = more uncertainty.
    """
    feat_acts = get_feature_activations(sae, hidden_last_token)

    # Get top-k most active features
    active_mask = feat_acts > 0.01
    active_indices = np.where(active_mask)[0]

    if len(active_indices) < 2:
        return 0.0

    # Sort by activation strength and take top-k
    if len(active_indices) > top_k:
        top_indices = active_indices[np.argsort(feat_acts[active_indices])[-top_k:]]
    else:
        top_indices = active_indices

    if len(top_indices) < 2:
        return 0.0

    # Get decoder vectors for top-k features
    # Shape: (n_active, hidden_dim)
    decoder_vecs = sae.W_dec[top_indices].detach().float().numpy()

    # Compute pairwise cosine similarities between decoder vectors
    # This measures how much the active concepts compete in representation space
    norms = np.linalg.norm(decoder_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = decoder_vecs / norms
    corr_matrix = normalized @ normalized.T

    upper_idx = np.triu_indices_from(corr_matrix, k=1)
    upper = corr_matrix[upper_idx]

    if upper.size == 0:
        return 0.0

    return float(np.percentile(upper, 5))


def run_exp3(
    model,
    tokenizer,
    ambiguous_prompts: list,
    exp1_results: dict,
    results_dir: str,
    debug: bool = False,
) -> dict:
    """
    Experiment 3 — Rivalry as an Uncertainty Signal.

    Compares whether the feature rivalry score predicts answer correctness better
    than softmax confidence (measured by AUROC).
    """
    from sklearn.metrics import roc_auc_score

    if debug:
        test_prompts = ambiguous_prompts[:4] + exp1_results.get("unambiguous_prompts", [])[:4]
    else:
        import json

        splits_path = os.path.join(results_dir, "dataset_splits.json")
        with open(splits_path) as f:
            splits = json.load(f)
        unambiguous_prompts = splits["unambiguous"]
        test_prompts = ambiguous_prompts + unambiguous_prompts

    peak_layer = max(
        exp1_results["ambiguous"].keys(),
        key=lambda l: (
            exp1_results["unambiguous"][l]["p5"]
            - exp1_results["ambiguous"][l]["p5"]
        ),
    )
    assert peak_layer in LAYERS_TO_ANALYZE
    print(f"Exp3 peak layer (max unambiguous p5 - ambiguous p5): {peak_layer}")
    sae = load_sae(peak_layer, SAE_RELEASE, SAE_IDS[peak_layer])

    checkpoint_path = os.path.join(results_dir, "exp3_checkpoint.pkl")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            exp3_results = pickle.load(f)
        completed_questions = {r["question"] for r in exp3_results}
        print(f"Resuming Exp 3: {len(completed_questions)} prompts already done")
    else:
        exp3_results = []
        completed_questions = set()

    for item in tqdm(test_prompts, desc="Exp3"):
        if item["question"] in completed_questions:
            continue

        prompt = f"Q: {item['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        with ActivationCache(model, [peak_layer], full_sequence=True) as cache:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                top_token_id = probs.argmax().item()
                softmax_conf = probs[top_token_id].item()

            hidden_seq = cache.get_activations(peak_layer)
            if hidden_seq is None:
                rivalry_score = 0.0
            else:
                last_token_hidden = hidden_seq[-1] if hidden_seq.ndim == 2 else hidden_seq[0, -1]
                rivalry_score = _per_prompt_rivalry_score(sae, last_token_hidden, top_k=50)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            gen_out[0][input_ids.shape[1]:].cpu(),
            skip_special_tokens=True,
        ).strip()

        ground_truth = item["answer"].lower().strip()
        is_correct = ground_truth in generated.lower()

        exp3_results.append(
            {
                "question": item["question"],
                "ground_truth": item["answer"],
                "generated": generated,
                "rivalry_score": rivalry_score,
                "softmax_conf": softmax_conf,
                "is_correct": is_correct,
            }
        )
        torch.cuda.empty_cache()

        # Checkpoint every 10 prompts
        if len(exp3_results) % 10 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(exp3_results, f)

    # ---- AUROC evaluation ----
    labels = [r["is_correct"] for r in exp3_results]
    # Negate rivalry (less negative = more confident = higher predicted correctness)
    rivalry_scores = [-r["rivalry_score"] for r in exp3_results]
    softmax_scores = [r["softmax_conf"] for r in exp3_results]

    if len(set(labels)) > 1:
        auroc_rivalry = roc_auc_score(labels, rivalry_scores)
        auroc_softmax = roc_auc_score(labels, softmax_scores)
        print(f"\nAUROC (rivalry score): {auroc_rivalry:.3f}")
        print(f"AUROC (softmax conf):  {auroc_softmax:.3f}")
        print(
            f"Rivalry {'BETTER' if auroc_rivalry > auroc_softmax else 'WORSE'} "
            "than softmax confidence"
        )
    else:
        print(
            "WARNING: all labels same — cannot compute AUROC. "
            "Need more varied prompts."
        )
        auroc_rivalry = auroc_softmax = None

    out = {
        "results": exp3_results,
        "auroc_rivalry": auroc_rivalry,
        "auroc_softmax": auroc_softmax,
    }

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp3_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"Exp 3 results saved to {out_path}")
    return out
