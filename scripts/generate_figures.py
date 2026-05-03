#!/usr/bin/env python3
"""
Regenerate paper figures from cached pickles (no GPU).

Requires: results/exp1_results.pkl, results/exp3_results.pkl
Optional: results/exp2_results.pkl (Exp 2 figures skipped if missing)

Each figure is written as .pdf and .png (see FIG_PNG_DPI).

Run from repo root:
  python scripts/generate_figures.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"
# PNG resolution (slides / web; bump to 300 for print)
FIG_PNG_DPI = 200


def _save_fig(fig: plt.Figure, stem: str) -> None:
    """Write PDF + PNG for the same figure stem (no extension)."""
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        FIGURES_DIR / f"{stem}.png",
        bbox_inches="tight",
        dpi=FIG_PNG_DPI,
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"Saved: {stem}.pdf, {stem}.png")

PATHS = {
    "exp1": RESULTS_DIR / "exp1_results.pkl",
    "exp2": RESULTS_DIR / "exp2_results.pkl",
    "exp3": RESULTS_DIR / "exp3_results.pkl",
}


def _require_exp1_exp3() -> None:
    missing = [str(PATHS[k]) for k in ("exp1", "exp3") if not PATHS[k].is_file()]
    if missing:
        print("Missing pickle(s):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    os.chdir(ROOT)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _require_exp1_exp3()

    with open(PATHS["exp1"], "rb") as f:
        exp1 = pickle.load(f)

    exp2 = None
    if PATHS["exp2"].is_file():
        with open(PATHS["exp2"], "rb") as f:
            exp2 = pickle.load(f)
    else:
        print(
            f"Note: {PATHS['exp2']} missing — skipping exp2_flip_rates and exp2_gap_vs_correlation",
            file=sys.stderr,
        )

    with open(PATHS["exp3"], "rb") as f:
        exp3_out = pickle.load(f)

    exp3 = exp3_out["results"]

    print("\nLayer-by-layer direction check:")
    layers_check = sorted(exp1["ambiguous"].keys())
    for l in layers_check:
        amb = exp1["ambiguous"][l]["p5"]
        unamb = exp1["unambiguous"][l]["p5"]
        p = exp1["stats"].get(l, {}).get("p_value", 1.0)
        direction = "AMB>UNAMB (correct)" if amb < unamb else "UNAMB>AMB (unexpected)"
        sig = "SIGNIFICANT" if p < 0.05 else ""
        print(f"Layer {l:2d}: amb={amb:.4f} unamb={unamb:.4f} | {direction} | p={p:.2e} {sig}")
    print()

    # ── Figure 1: Rivalry by Layer ───────────────────────────────────────────
    layers = sorted(exp1["ambiguous"].keys())
    amb_p5 = [exp1["ambiguous"][l]["p5"] for l in layers]
    unamb_p5 = [exp1["unambiguous"][l]["p5"] for l in layers]
    p_values = [exp1["stats"].get(l, {}).get("p_value", 1.0) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        layers,
        amb_p5,
        color="red",
        marker="o",
        linewidth=2,
        label="Ambiguous (high entropy)",
    )
    ax.plot(
        layers,
        unamb_p5,
        color="blue",
        marker="o",
        linewidth=2,
        label="Unambiguous (low entropy)",
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    for i, (layer, p) in enumerate(zip(layers, p_values)):
        if p < 0.05:
            ax.axvline(layer, color="green", linestyle=":", alpha=0.6)
            ax.annotate(
                f"p={p:.1e}",
                xy=(layer, min(amb_p5[i], unamb_p5[i]) - 0.02),
                fontsize=8,
                color="green",
                ha="center",
            )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("5th Percentile Pairwise Correlation (Rivalry)", fontsize=12)
    ax.set_title(
        "Feature Rivalry by Layer: Ambiguous vs Unambiguous Inputs", fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "rivalry_by_layer")

    # ── Figure 2: P-value by Layer (direction-aware) ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))

    for layer, p in zip(layers, p_values):
        log_p = -np.log10(max(p, 1e-30))
        amb_val = exp1["ambiguous"][layer]["p5"]
        unamb_val = exp1["unambiguous"][layer]["p5"]
        correct_direction = amb_val < unamb_val  # ambiguous more negative = more rivalry

        if p < 0.05 and correct_direction:
            color = "green"
        elif p < 0.05 and not correct_direction:
            color = "red"
        else:
            color = "lightgray"

        ax.bar(layer, log_p, color=color, width=1.5)

    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("-log10(p-value)", fontsize=12)
    ax.set_title(
        "Statistical Significance of Rivalry Difference by Layer", fontsize=13
    )

    green_patch = mpatches.Patch(
        color="green",
        label="Significant: ambiguous > unambiguous rivalry (correct direction)",
    )
    red_patch = mpatches.Patch(
        color="red",
        label="Significant: unambiguous > ambiguous rivalry (unexpected direction)",
    )
    ns_patch = mpatches.Patch(color="lightgray", label="Not significant")
    threshold_line = mlines.Line2D(
        [0], [0], color="black", linestyle="--", label="p=0.05"
    )
    ax.legend(
        handles=[green_patch, red_patch, ns_patch, threshold_line], fontsize=9
    )
    ax.grid(True, alpha=0.3, axis="y")
    _save_fig(fig, "pvalue_by_layer")

    if exp2 is not None:
        # ── Figure 3: Exp 2 Flip Rate Comparison at mult=5 ─────────────────
        mult_target = 5
        pairs_at_mult = [r for r in exp2 if r["multiplier"] == mult_target]
        pairs_at_mult = sorted(pairs_at_mult, key=lambda r: r["correlation"])

        pair_labels = [
            f"({r['feature_i']},{r['feature_j']})\n{r['correlation']:.3f}"
            for r in pairs_at_mult
        ]
        flip_B = [r["flip_rate_B"] for r in pairs_at_mult]
        flip_rand = [r["flip_rate_random"] for r in pairs_at_mult]

        x = np.arange(len(pairs_at_mult))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(
            x - width / 2,
            flip_B,
            width,
            label="Rivalry Axis Steering",
            color="steelblue",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            flip_rand,
            width,
            label="Random Steering",
            color="lightcoral",
            alpha=0.85,
        )

        ax.set_xlabel("Feature Pair (sorted by correlation strength)", fontsize=11)
        ax.set_ylabel("Flip Rate", fontsize=12)
        ax.set_title(
            f"Causal Intervention: Rivalry vs Random Steering (multiplier={mult_target})",
            fontsize=13,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=7, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        _save_fig(fig, "exp2_flip_rates")

        # ── Figure 4: Exp 2 Gap vs Correlation Strength ─────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        for mult, color, marker in [
            (5, "steelblue", "o"),
            (10, "orange", "s"),
            (20, "green", "^"),
        ]:
            pairs = [r for r in exp2 if r["multiplier"] == mult]
            corrs = [r["correlation"] for r in pairs]
            gaps = [r["flip_rate_B"] - r["flip_rate_random"] for r in pairs]
            ax.scatter(
                corrs,
                gaps,
                color=color,
                marker=marker,
                label=f"mult={mult}",
                alpha=0.7,
                s=60,
            )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Pairwise Feature Correlation (Rivalry Strength)", fontsize=12)
        ax.set_ylabel("Flip Rate Gap (Rivalry - Random)", fontsize=12)
        ax.set_title("Causal Influence vs Rivalry Strength", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        _save_fig(fig, "exp2_gap_vs_correlation")

    # ── Figure 5: ROC Curve (Exp 3) ──────────────────────────────────────────
    labels = [int(r["is_correct"]) for r in exp3]
    rivalry_scores = [
        -(r["rivalry_score"] if "rivalry_score" in r else r["peak_rivalry"])
        for r in exp3
    ]
    softmax_scores = [r["softmax_conf"] for r in exp3]

    fpr_r, tpr_r, _ = roc_curve(labels, rivalry_scores)
    fpr_s, tpr_s, _ = roc_curve(labels, softmax_scores)
    auroc_r = auc(fpr_r, tpr_r)
    auroc_s = auc(fpr_s, tpr_s)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr_r,
        tpr_r,
        color="steelblue",
        linewidth=2,
        label=f"Rivalry Score (AUROC={auroc_r:.3f})",
    )
    ax.plot(
        fpr_s,
        tpr_s,
        color="orange",
        linewidth=2,
        label=f"Softmax Confidence (AUROC={auroc_s:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Random (AUROC=0.500)",
    )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve: Rivalry Score vs Softmax Confidence", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "roc_curve")

    # ── Figure 6: Calibration ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, scores, name, color in zip(
        axes,
        [rivalry_scores, softmax_scores],
        ["Rivalry Score", "Softmax Confidence"],
        ["steelblue", "orange"],
    ):
        bins = np.linspace(min(scores), max(scores), 6)
        bin_acc = []
        bin_centers = []
        bin_counts = []

        for i in range(len(bins) - 1):
            mask = [bins[i] <= s < bins[i + 1] for s in scores]
            if sum(mask) > 0:
                acc = np.mean([labels[j] for j, m in enumerate(mask) if m])
                bin_acc.append(acc)
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_counts.append(sum(mask))

        ax.bar(
            bin_centers,
            bin_acc,
            width=(bins[1] - bins[0]) * 0.8,
            alpha=0.8,
            color=color,
        )
        for center, acc, count in zip(bin_centers, bin_acc, bin_counts):
            ax.text(center, acc + 0.02, f"n={count}", ha="center", fontsize=8)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Calibration: {name}", fontsize=13)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    _save_fig(fig, "calibration")

    print("\nAll figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
