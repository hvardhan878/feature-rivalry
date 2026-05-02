import os
import pickle

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from analysis.rivalry_score import find_top_rival_pairs
from config import SAE_RELEASE, SAE_IDS
from model.hooks import ActivationCache
from sae.gemma_scope import get_feature_activations, load_sae


def run_exp1(
    model,
    tokenizer,
    ambiguous_prompts: list,
    unambiguous_prompts: list,
    results_dir: str,
    layers_to_analyze: list,
    debug: bool = False,
) -> dict:
    """
    Experiment 1 — Existence of Feature Rivalry.

    For each condition (ambiguous / unambiguous), collect SAE feature activations
    at every layer, compute pairwise correlations, and test whether ambiguous inputs
    produce more negatively correlated feature pairs (Mann-Whitney U).
    """
    results: dict = {"ambiguous": {}, "unambiguous": {}}

    for condition in ["ambiguous", "unambiguous"]:
        prompts = ambiguous_prompts if condition == "ambiguous" else unambiguous_prompts
        checkpoint_path = os.path.join(results_dir, f"exp1_{condition}_checkpoint.pkl")

        # Resume from checkpoint if available
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                layer_feature_matrix = pickle.load(f)
            start_idx = min(len(v) for v in layer_feature_matrix.values())
            print(f"Resuming {condition} from prompt {start_idx}")
        else:
            layer_feature_matrix = {layer: [] for layer in layers_to_analyze}
            start_idx = 0

        for i, item in enumerate(
            tqdm(prompts[start_idx:], desc=f"Exp1 {condition}"),
            start=start_idx,
        ):
            prompt = f"Q: {item['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to("mps")
            attention_mask = inputs["attention_mask"].to("mps")

            with ActivationCache(model, layers_to_analyze) as cache:
                with torch.no_grad():
                    model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                for layer in layers_to_analyze:
                    hidden = cache.get_activations(layer)
                    if hidden is None:
                        continue
                    sae = load_sae(layer, SAE_RELEASE, SAE_IDS[layer])
                    features = get_feature_activations(sae, hidden)
                    layer_feature_matrix[layer].append(features)

            torch.mps.empty_cache()

            # Save checkpoint every 100 prompts
            if (i + 1) % 100 == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(layer_feature_matrix, f)
                print(f"  Checkpoint saved at prompt {i + 1}")

        # ---- Per-layer correlation analysis ----
        print(f"\nComputing correlations for {condition}...")
        for layer in layers_to_analyze:
            matrix = np.array(layer_feature_matrix[layer])  # (n_prompts, 16384)

            # Keep only features that are active on average
            active_mask = matrix.mean(axis=0) > 0.01
            active_indices = np.where(active_mask)[0]

            # Memory safety: subsample if too many active features
            if len(active_indices) > 300:
                np.random.seed(42)
                active_indices = np.random.choice(active_indices, 300, replace=False)

            active_matrix = matrix[:, active_indices]  # (n_prompts, n_active)

            if active_matrix.shape[1] < 2:
                print(f"  Layer {layer}: too few active features, skipping")
                continue

            corr_matrix = np.corrcoef(active_matrix.T)  # (n_active, n_active)
            upper_idx = np.triu_indices_from(corr_matrix, k=1)
            upper = corr_matrix[upper_idx]

            results[condition][layer] = {
                "correlations": upper,
                "mean": float(upper.mean()),
                "p5": float(np.percentile(upper, 5)),
                "p25": float(np.percentile(upper, 25)),
                "active_indices": active_indices,
                "corr_matrix": corr_matrix,
                "n_active_features": len(active_indices),
            }

    # ---- Statistical test: Mann-Whitney U per layer ----
    stat_results: dict = {}
    print("\nLayer | Amb p5  | Unamb p5 | p-value")
    print("-" * 45)
    for layer in layers_to_analyze:
        if layer not in results["ambiguous"] or layer not in results["unambiguous"]:
            continue
        amb_corr = results["ambiguous"][layer]["correlations"]
        unamb_corr = results["unambiguous"][layer]["correlations"]
        u_stat, p_val = stats.mannwhitneyu(amb_corr, unamb_corr, alternative="less")
        stat_results[layer] = {"u_stat": float(u_stat), "p_value": float(p_val)}
        print(
            f"  {layer:2d}  | {results['ambiguous'][layer]['p5']:6.3f}  | "
            f"{results['unambiguous'][layer]['p5']:8.3f} | {p_val:.2e}"
        )

    results["stats"] = stat_results

    # ---- Save ----
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp1_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nExp 1 results saved to {out_path}")
    return results
