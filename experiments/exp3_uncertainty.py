import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from config import LAYERS_TO_ANALYZE, SAE_IDS, SAE_RELEASE
from model.hooks import ActivationCache
from sae.gemma_scope import get_feature_activations, load_sae


def _per_prompt_rivalry_score(sae, hidden_last_token: np.ndarray) -> float:
    """
    Compute rivalry score from last token hidden state only.
    Uses entropy of active SAE feature activations as proxy for feature competition.
    Higher entropy = more features competing = more rivalry.
    """
    feat_acts = get_feature_activations(sae, hidden_last_token)
    active = feat_acts[feat_acts > 0.01]
    if len(active) == 0:
        return 0.0
    probs = active / active.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


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
                rivalry_score = _per_prompt_rivalry_score(sae, last_token_hidden)

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
