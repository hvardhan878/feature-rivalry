import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from analysis.rivalry_score import compute_rivalry_score
from config import LAYERS_TO_ANALYZE, SAE_IDS, SAE_RELEASE
from model.hooks import ActivationCache
from sae.gemma_scope import get_feature_activations, load_sae


def _per_prompt_rivalry_from_seq_hidden(sae, hidden_seq: np.ndarray) -> float:
    """
    hidden_seq: (seq_len, hidden_dim) last-layer hidden states for one prompt.
    Returns 5th percentile of pairwise correlations of active SAE features (max act > 0.01),
    correlating across sequence positions (same construction as Exp 1, one prompt).
    """
    seq_len = hidden_seq.shape[0]
    if seq_len < 2:
        return 0.0

    feat_rows = []
    for t in range(seq_len):
        feat_rows.append(get_feature_activations(sae, hidden_seq[t]))
    matrix = np.stack(feat_rows, axis=0)  # (seq_len, n_features)

    active_mask = matrix.max(axis=0) > 0.01
    active_indices = np.where(active_mask)[0]

    if len(active_indices) > 300:
        rng = np.random.default_rng(42)
        active_indices = rng.choice(active_indices, 300, replace=False)

    if len(active_indices) < 2:
        return 0.0

    active_matrix = matrix[:, active_indices]
    corr_matrix = np.corrcoef(active_matrix.T)
    upper_idx = np.triu_indices_from(corr_matrix, k=1)
    upper = corr_matrix[upper_idx]
    if upper.size == 0:
        return 0.0
    return compute_rivalry_score(upper)


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

    test_prompts = ambiguous_prompts[:8] if debug else ambiguous_prompts

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
                rivalry_score = _per_prompt_rivalry_from_seq_hidden(sae, hidden_seq)

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
