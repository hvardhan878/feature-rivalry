import os
import pickle

import torch
from tqdm import tqdm

from analysis.rivalry_score import compute_rivalry_score


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

    # Pre-compute per-layer rivalry scores from Exp 1 (static across prompts)
    rivalry_by_layer = {
        layer: compute_rivalry_score(exp1_results["ambiguous"][layer]["correlations"])
        for layer in exp1_results["ambiguous"]
    }
    peak_rivalry = min(rivalry_by_layer.values())  # most negative = strongest rivalry

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
        input_ids = inputs["input_ids"].to("mps")
        attention_mask = inputs["attention_mask"].to("mps")

        with torch.no_grad():
            # Logits for next-token softmax confidence
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_token_id = probs.argmax().item()
            softmax_conf = probs[top_token_id].item()

            # Greedy answer generation
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
                "peak_rivalry": peak_rivalry,
                "softmax_conf": softmax_conf,
                "is_correct": is_correct,
            }
        )
        torch.mps.empty_cache()

        # Checkpoint every 10 prompts
        if len(exp3_results) % 10 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(exp3_results, f)

    # ---- AUROC evaluation ----
    labels = [r["is_correct"] for r in exp3_results]
    # Negate rivalry (less negative = more confident = higher predicted correctness)
    rivalry_scores = [-r["peak_rivalry"] for r in exp3_results]
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
