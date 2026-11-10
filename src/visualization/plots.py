"""
Scientific Plotting and Research Visualization Module.
Generates publication-quality charts for class distributions, preprocessing comparisons,
model prediction confidence, claim verification breakdowns, and calibration diagrams.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_class_distributions(df: pd.DataFrame, target_classes: List[str]) -> plt.Figure:
    """Plots frequency distribution of target abnormalities in the dataset."""
    counts = {c: 0 for c in target_classes}

    for _, row in df.iterrows():
        raw_labels = row.get("labels", "")
        if isinstance(raw_labels, str):
            lbl_list = [l.strip() for l in raw_labels.replace(";", ",").split(",") if l.strip()]
        elif isinstance(raw_labels, list):
            lbl_list = raw_labels
        else:
            lbl_list = []

        for lbl in lbl_list:
            for target in target_classes:
                if target.lower() in lbl.lower():
                    counts[target] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    classes = list(counts.keys())
    values = list(counts.values())

    bars = ax.barh(classes, values, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_title("Dataset Abnormality Class Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_preprocessing_comparison(stages: Dict[str, Image.Image]) -> plt.Figure:
    """Displays comparative visual progression through preprocessing pipeline."""
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    if len(stages) == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, stages.items()):
        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_model_predictions(predictions: Dict[str, float], thresholds: Optional[Dict[str, float]] = None) -> plt.Figure:
    """Visualizes predicted probabilities across target abnormalities with decision thresholds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=False)
    classes = [k for k, v in sorted_items]
    probs = [v for k, v in sorted_items]

    # Color code by threshold
    colors = ["#2ca02c" if p >= 0.60 else "#ff7f0e" if p >= 0.25 else "#1f77b4" for p in probs]

    bars = ax.barh(classes, probs, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Model Probability Output", fontsize=11, fontweight="bold")
    ax.set_title("Chest Radiograph Multi-Label Probabilities", fontsize=13, fontweight="bold")

    # Threshold guidelines
    ax.axvline(0.60, color="#d62728", linestyle="--", linewidth=1.5, label="Supported Threshold (0.60)")
    ax.axvline(0.20, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Contradiction Threshold (0.20)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_claim_verification_summary(verification_results: List[Any]) -> plt.Figure:
    """Visualizes distribution of claim verification statuses."""
    status_counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNCERTAIN": 0, "CONTRADICTED": 0}
    for r in verification_results:
        s = getattr(r, "status", "UNCERTAIN")
        status_counts[s] = status_counts.get(s, 0) + 1

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(status_counts.keys())
    counts = list(status_counts.values())
    colors = ["#2ca02c", "#ff7f0e", "#7f7f7f", "#d62728"]

    bars = ax.bar(labels, counts, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Count of Verified Claims", fontsize=11, fontweight="bold")
    ax.set_title("Independent Claim Verification Breakdown", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{int(h)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_experiment_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Plots comparative text fidelity vs hallucination rates across experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Metric 1: Clinical F1 and BLEU-2
    x = np.arange(len(results_df))
    width = 0.35

    ax1.bar(x - width/2, results_df["Clinical_F1"], width, label="Clinical F1", color="#1f77b4", edgecolor="black")
    ax1.bar(x + width/2, results_df["BLEU-2"], width, label="BLEU-2", color="#aec7e8", edgecolor="black")
    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_title("Report Generation & Clinical Factuality", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df["Experiment"], rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Metric 2: Unsupported Claim Rate and Contradiction Rate (Lower is Better)
    ax2.bar(x - width/2, results_df["Unsupported_Claim_Rate"], width, label="Unsupported Rate (Lower=Better)", color="#ff7f0e", edgecolor="black")
    ax2.bar(x + width/2, results_df["Contradiction_Rate"], width, label="Contradiction Rate (Lower=Better)", color="#d62728", edgecolor="black")
    ax2.set_ylabel("Error Rate", fontsize=11, fontweight="bold")
    ax2.set_title("Hallucination & Contradiction Reduction", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df["Experiment"], rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig


def plot_reliability_diagram(ece: float, bin_data: Dict[str, Any]) -> plt.Figure:
    """Plots empirical reliability diagram comparing confidence against accuracy."""
    fig, ax = plt.subplots(figsize=(6, 6))

    bins = bin_data.get("bins", [])
    confs = [b["confidence"] for b in bins if b["size"] > 0]
    accs = [b["accuracy"] for b in bins if b["size"] > 0]

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    if confs and accs:
        ax.plot(confs, accs, marker="o", color="#1f77b4", linewidth=2, label=f"Model (ECE={ece:.4f})")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig

