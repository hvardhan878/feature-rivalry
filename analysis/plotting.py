import os

import matplotlib.pyplot as plt
import numpy as np


def plot_rivalry_by_layer(exp1_results: dict, results_dir: str) -> None:
    """
    Plot 5th-percentile pairwise correlation (rivalry) by layer for both conditions.
    """
    layers = sorted(exp1_results["ambiguous"].keys())
    amb_p5 = [exp1_results["ambiguous"][l]["p5"] for l in layers]
    unamb_p5 = [exp1_results["unambiguous"][l]["p5"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, amb_p5, color="red", marker="o", label="Ambiguous")
    ax.plot(layers, unamb_p5, color="blue", marker="o", label="Unambiguous")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("5th Percentile Pairwise Correlation (Rivalry)")
    ax.set_title("Feature Rivalry by Layer: Ambiguous vs Unambiguous Inputs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    path = os.path.join(results_dir, "figures", "rivalry_by_layer.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_calibration(exp3_out: dict, results_dir: str) -> None:
    """
    Plot calibration curves comparing rivalry score vs softmax confidence as
    predictors of answer correctness.
    """
    results = exp3_out["results"]
    rivalry_scores = [
        -r["rivalry_score"] if "rivalry_score" in r else -r["peak_rivalry"]
        for r in results
    ]
    softmax_scores = [r["softmax_conf"] for r in results]
    labels = [int(r["is_correct"]) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, scores, name in zip(
        axes,
        [rivalry_scores, softmax_scores],
        ["Rivalry Score", "Softmax Confidence"],
    ):
        bins = np.linspace(min(scores), max(scores), 6)
        bin_acc = []
        bin_centers = []

        for i in range(len(bins) - 1):
            mask = [bins[i] <= s < bins[i + 1] for s in scores]
            if sum(mask) > 0:
                acc = np.mean([labels[j] for j, m in enumerate(mask) if m])
                bin_acc.append(acc)
                bin_centers.append((bins[i] + bins[i + 1]) / 2)

        ax.bar(
            bin_centers,
            bin_acc,
            width=(bins[1] - bins[0]) * 0.8,
            alpha=0.7,
        )
        ax.set_xlabel(name)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Calibration: {name}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    path = os.path.join(results_dir, "figures", "calibration.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
