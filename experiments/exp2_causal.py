import os
import pickle
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from analysis.rivalry_score import find_top_rival_pairs
from config import HIDDEN_DIM, SAE_RELEASE, SAE_IDS, TOP_K_RIVAL_PAIRS

STEERING_MULTIPLIERS = [5, 10, 20]
from sae.gemma_scope import load_sae


def run_exp2(
    model,
    tokenizer,
    ambiguous_prompts: list,
    exp1_results: dict,
    results_dir: str,
    debug: bool = False,
) -> list:
    """
    Experiment 2 — Causal Test via Activation Steering.

    For the top rival feature pairs identified in Exp 1, steer the residual stream
    at the peak rivalry layer by adding the decoder direction of feature B, then
    measure how often the model's output changes compared to baseline (and versus
    a random steering direction), for steering multipliers 5, 10, and 20.
    """
    # Layer where ambiguous rivalry is strongest relative to unambiguous:
    # max_l (unambiguous p5 - ambiguous p5)
    peak_layer = max(
        exp1_results["ambiguous"].keys(),
        key=lambda l: (
            exp1_results["unambiguous"][l]["p5"]
            - exp1_results["ambiguous"][l]["p5"]
        ),
    )
    print(f"Peak rivalry layer: {peak_layer}")

    top_pairs = find_top_rival_pairs(
        exp1_results["ambiguous"][peak_layer]["corr_matrix"],
        exp1_results["ambiguous"][peak_layer]["active_indices"],
        TOP_K_RIVAL_PAIRS,
    )

    sae = load_sae(peak_layer, SAE_RELEASE, SAE_IDS[peak_layer])

    pairs_to_test = top_pairs[:2] if debug else top_pairs[:20]
    test_prompts = ambiguous_prompts[:4] if debug else ambiguous_prompts[:50]

    checkpoint_path = os.path.join(results_dir, "exp2_checkpoint.pkl")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        exp2_results = checkpoint["results"]
        pair_counts = Counter(
            (r["feature_i"], r["feature_j"]) for r in exp2_results
        )
        n_mult = len(STEERING_MULTIPLIERS)
        n_done_pairs = sum(1 for c in pair_counts.values() if c >= n_mult)
        print(f"Resuming Exp 2: {n_done_pairs} pairs already done")
    else:
        exp2_results = []

    for pair in tqdm(pairs_to_test, desc="Exp2 pairs"):
        feat_A = pair["feature_i"]
        feat_B = pair["feature_j"]

        pair_rows = [
            r
            for r in exp2_results
            if r["feature_i"] == feat_A and r["feature_j"] == feat_B
        ]
        mults_done = {r.get("multiplier") for r in pair_rows}
        if mults_done == set(STEERING_MULTIPLIERS):
            continue
        if pair_rows:
            exp2_results = [
                r
                for r in exp2_results
                if not (r["feature_i"] == feat_A and r["feature_j"] == feat_B)
            ]

        # Decoder direction for feature B — the "concept vector" in residual-stream space
        vec_B = sae.W_dec[feat_B].detach().float().numpy()  # (hidden_dim,)

        # Random baseline direction (unit-normalised)
        random_vecs = [np.random.randn(HIDDEN_DIM).astype(np.float32) for _ in range(10)]
        random_vecs = [v / np.linalg.norm(v) for v in random_vecs]
        vec_random = np.mean(random_vecs, axis=0).astype(np.float32)
        vec_random = vec_random / np.linalg.norm(vec_random)

        def make_steer_hook(vec_numpy, multiplier: float):
            """Return a forward hook that adds a steering vector to the residual stream."""
            vec_tensor = torch.from_numpy(vec_numpy).float().to("cuda")

            def hook(module, input, output):
                steered = output[0].clone()
                steered[0, :, :] += multiplier * vec_tensor
                return (steered,) + output[1:]

            return hook

        for mult in STEERING_MULTIPLIERS:
            flip_count_B = 0
            flip_count_random = 0
            total = 0

            for item in test_prompts:
                prompt = f"Q: {item['question']}\nA:"
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to("cuda")
                attention_mask = inputs["attention_mask"].to("cuda")

                # --- Baseline generation ---
                with torch.no_grad():
                    baseline_out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                baseline_text = tokenizer.decode(
                    baseline_out[0][input_ids.shape[1]:].cpu(),
                    skip_special_tokens=True,
                ).strip()

                # --- Steered-B generation ---
                handle = model.model.layers[peak_layer].register_forward_hook(
                    make_steer_hook(vec_B, mult)
                )
                try:
                    with torch.no_grad():
                        steered_B_out = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=10,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                finally:
                    handle.remove()

                steered_B_text = tokenizer.decode(
                    steered_B_out[0][input_ids.shape[1]:].cpu(),
                    skip_special_tokens=True,
                ).strip()

                # --- Steered-random generation ---
                handle = model.model.layers[peak_layer].register_forward_hook(
                    make_steer_hook(vec_random, mult)
                )
                try:
                    with torch.no_grad():
                        steered_rand_out = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=10,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                finally:
                    handle.remove()

                steered_rand_text = tokenizer.decode(
                    steered_rand_out[0][input_ids.shape[1]:].cpu(),
                    skip_special_tokens=True,
                ).strip()

                flipped_B = baseline_text.lower() != steered_B_text.lower()
                flipped_rand = baseline_text.lower() != steered_rand_text.lower()
                flip_count_B += int(flipped_B)
                flip_count_random += int(flipped_rand)
                total += 1
                torch.cuda.empty_cache()

            result = {
                "feature_i": feat_A,
                "feature_j": feat_B,
                "correlation": pair["correlation"],
                "multiplier": mult,
                "flip_rate_B": flip_count_B / total,
                "flip_rate_random": flip_count_random / total,
                "total_prompts": total,
            }
            exp2_results.append(result)
            print(
                f"  Pair ({feat_A},{feat_B}) mult={mult} corr={pair['correlation']:.3f} | "
                f"flip_B={result['flip_rate_B']:.2f}  flip_rand={result['flip_rate_random']:.2f}"
            )

        # Checkpoint after every pair (pairs are slow)
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"results": exp2_results}, f)

    # Final save + remove checkpoint on clean completion
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp2_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(exp2_results, f)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"\nExp 2 results saved to {out_path}")
    return exp2_results
