import argparse
import json
import os
import pickle
import sys

# Ensure project root is on path when invoked from a different working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import config as _cfg
from config import (
    LAYERS_TO_ANALYZE,
    MODEL_NAME,
    N_PROMPTS_PER_CONDITION,
    N_SAMPLES_FOR_ENTROPY,
    RESULTS_DIR,
)
from model.loader import load_model_and_tokenizer
from data.dataset_builder import (
    load_unambiguous_sample,
    load_pop_qa_sample,
    build_ambiguous_unambiguous_split,
)
from experiments.exp1_existence import run_exp1
from experiments.exp2_causal import run_exp2
from experiments.exp3_uncertainty import run_exp3
from analysis.plotting import plot_rivalry_by_layer, plot_calibration


# ── Canonical result / checkpoint paths ─────────────────────────────────────

def _paths(results_dir: str) -> dict:
    d = results_dir
    return {
        "dataset":       os.path.join(d, "dataset_splits.json"),
        "exp1":          os.path.join(d, "exp1_results.pkl"),
        "exp1_ckpt_amb": os.path.join(d, "exp1_ambiguous_checkpoint.pkl"),
        "exp1_ckpt_un":  os.path.join(d, "exp1_unambiguous_checkpoint.pkl"),
        "exp2":          os.path.join(d, "exp2_results.pkl"),
        "exp2_ckpt":     os.path.join(d, "exp2_checkpoint.pkl"),
        "exp3":          os.path.join(d, "exp3_results.pkl"),
        "exp3_ckpt":     os.path.join(d, "exp3_checkpoint.pkl"),
    }


# ── --status implementation ──────────────────────────────────────────────────

def print_status(results_dir: str) -> None:
    p = _paths(results_dir)

    print()
    print("=" * 50)
    print("FEATURE RIVALRY — PIPELINE STATUS")
    print("=" * 50)

    # Dataset split
    if os.path.exists(p["dataset"]):
        with open(p["dataset"]) as f:
            splits = json.load(f)
        n_amb = len(splits.get("ambiguous", []))
        n_un  = len(splits.get("unambiguous", []))
        print(f"Dataset split:     DONE ({n_amb} ambiguous, {n_un} unambiguous)")
    else:
        print("Dataset split:     NOT STARTED")

    # Experiment 1
    if os.path.exists(p["exp1"]):
        print("Exp 1:             DONE (results cached)")
    else:
        # Check whether either condition checkpoint exists
        has_amb = os.path.exists(p["exp1_ckpt_amb"])
        has_un  = os.path.exists(p["exp1_ckpt_un"])
        if has_amb or has_un:
            counts = []
            for label, ckpt_path in [("ambiguous", p["exp1_ckpt_amb"]),
                                      ("unambiguous", p["exp1_ckpt_un"])]:
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "rb") as f:
                        ckpt = pickle.load(f)
                    n_done = min(len(v) for v in ckpt.values()) if ckpt else 0
                    counts.append(f"{label}: {n_done} prompts")
            print(f"Exp 1:             IN PROGRESS ({', '.join(counts)})")
        else:
            print("Exp 1:             NOT STARTED")

    # Experiment 2
    if os.path.exists(p["exp2"]):
        with open(p["exp2"], "rb") as f:
            exp2_data = pickle.load(f)
        print(f"Exp 2:             DONE ({len(exp2_data)} pairs)")
    elif os.path.exists(p["exp2_ckpt"]):
        with open(p["exp2_ckpt"], "rb") as f:
            ckpt = pickle.load(f)
        n_done = len(ckpt.get("results", []))
        print(f"Exp 2:             IN PROGRESS ({n_done} pairs completed)")
    else:
        print("Exp 2:             NOT STARTED")

    # Experiment 3
    if os.path.exists(p["exp3"]):
        with open(p["exp3"], "rb") as f:
            exp3_data = pickle.load(f)
        n = len(exp3_data.get("results", []))
        auroc = exp3_data.get("auroc_rivalry")
        auroc_str = f", AUROC rivalry={auroc:.3f}" if auroc is not None else ""
        print(f"Exp 3:             DONE ({n} prompts{auroc_str})")
    elif os.path.exists(p["exp3_ckpt"]):
        with open(p["exp3_ckpt"], "rb") as f:
            ckpt = pickle.load(f)
        n_done = len(ckpt) if isinstance(ckpt, list) else 0
        print(f"Exp 3:             IN PROGRESS ({n_done} prompts completed)")
    else:
        print("Exp 3:             NOT STARTED")

    print("=" * 50)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature Rivalry Experiment Runner")
    parser.add_argument(
        "--exp",
        choices=["all", "1", "2", "3"],
        default="all",
        help="Which experiment(s) to run",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use minimal data for a fast pipeline check (~2–5 min)",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Load dataset from cache, skip entropy measurement",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print pipeline progress summary and exit (no computation)",
    )
    args = parser.parse_args()

    if args.status:
        print_status(RESULTS_DIR)
        return

    # ── Debug overrides ──────────────────────────────────────────────────────
    if args.debug:
        print("=" * 50)
        print("DEBUG MODE: minimal data, fast pipeline check")
        print("=" * 50)
        _cfg.N_PROMPTS_PER_CONDITION = 4
        _cfg.N_SAMPLES_FOR_ENTROPY = 3
        _cfg.LAYERS_TO_ANALYZE = [12]  # single mid-network layer
        # With n_samples=3, normalized entropy is only 0.0, 0.835, or 1.0.
        # Loosen thresholds so the pipeline has enough questions in both buckets.
        _cfg.ENTROPY_THRESHOLD_HIGH = 0.5   # captures 2-1 split and all-different
        _cfg.ENTROPY_THRESHOLD_LOW = 0.9    # captures all-same and 2-1 split
        n_prompts = _cfg.N_PROMPTS_PER_CONDITION
        layers    = _cfg.LAYERS_TO_ANALYZE
        n_to_load = 12
    else:
        n_prompts = N_PROMPTS_PER_CONDITION
        layers    = LAYERS_TO_ANALYZE

    if "n_to_load" not in dir():
        n_to_load = 2000

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    p = _paths(RESULTS_DIR)

    # ── Step 1: Load model ───────────────────────────────────────────────────
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # ── Step 2: Build dataset ────────────────────────────────────────────────
    print("\nBuilding dataset splits...")
    unambiguous_candidates = load_unambiguous_sample(n_to_load)
    popqa = load_pop_qa_sample(n_to_load)
    ambiguous, unambiguous = build_ambiguous_unambiguous_split(
        model,
        tokenizer,
        unambiguous_candidates,
        popqa,
        n_prompts,
        _cfg.ENTROPY_THRESHOLD_HIGH,
        _cfg.ENTROPY_THRESHOLD_LOW,
        RESULTS_DIR,
    )
    print(f"Dataset ready: {len(ambiguous)} ambiguous, {len(unambiguous)} unambiguous")

    # ── Experiment 1 ─────────────────────────────────────────────────────────
    exp1_results = None

    if args.exp in ["all", "1"]:
        print("\n" + "=" * 50)
        print("EXPERIMENT 1: Existence of Feature Rivalry")
        print("=" * 50)
        if os.path.exists(p["exp1"]):
            print(f"Loading cached Exp 1 results from {p['exp1']}")
            with open(p["exp1"], "rb") as f:
                exp1_results = pickle.load(f)
        else:
            exp1_results = run_exp1(
                model,
                tokenizer,
                ambiguous,
                unambiguous,
                RESULTS_DIR,
                layers,
                debug=args.debug,
            )
        plot_rivalry_by_layer(exp1_results, RESULTS_DIR)

    # ── Experiment 2 ─────────────────────────────────────────────────────────
    if args.exp in ["all", "2"]:
        print("\n" + "=" * 50)
        print("EXPERIMENT 2: Causal Test (Activation Steering)")
        print("=" * 50)
        if os.path.exists(p["exp2"]):
            print(f"Loading cached Exp 2 results from {p['exp2']}")
            with open(p["exp2"], "rb") as f:
                exp2_results = pickle.load(f)  # noqa: F841 — cached, no further use here
        else:
            if exp1_results is None:
                assert os.path.exists(p["exp1"]), "Run Exp 1 first before Exp 2"
                with open(p["exp1"], "rb") as f:
                    exp1_results = pickle.load(f)
            run_exp2(
                model,
                tokenizer,
                ambiguous,
                exp1_results,
                RESULTS_DIR,
                debug=args.debug,
            )

    # ── Experiment 3 ─────────────────────────────────────────────────────────
    if args.exp in ["all", "3"]:
        print("\n" + "=" * 50)
        print("EXPERIMENT 3: Rivalry as Uncertainty Signal")
        print("=" * 50)
        if os.path.exists(p["exp3"]):
            print(f"Loading cached Exp 3 results from {p['exp3']}")
            with open(p["exp3"], "rb") as f:
                exp3_out = pickle.load(f)
        else:
            if exp1_results is None:
                assert os.path.exists(p["exp1"]), "Run Exp 1 first before Exp 3"
                with open(p["exp1"], "rb") as f:
                    exp1_results = pickle.load(f)
            exp3_out = run_exp3(
                model,
                tokenizer,
                ambiguous,
                exp1_results,
                RESULTS_DIR,
                debug=args.debug,
            )
        plot_calibration(exp3_out, RESULTS_DIR)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if args.debug:
        print("DEBUG RUN COMPLETE — pipeline is working")
        print("Next step: python run.py --exp 1  (full run, ~8hrs overnight)")
    else:
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {RESULTS_DIR}")
        print(f"Figures saved to: {RESULTS_DIR}figures/")
    print("=" * 50)


if __name__ == "__main__":
    main()
