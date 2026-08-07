"""
 Visualizations for the fraud detection experiment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_arch(save_path="fig1_nn_arch.png"):
    fig, axes = plt.subplots(figsize=(10, 6))
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)
    axes.axis('off')

    layers = {
        "Input\n(30 features)": (1.5, [2, 3, 4, 5, 6, 7, 8]),
        "Hidden\n(64 neurons)": (5, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "Output\n(1 neuron)": (8.5, [5]),
    }


    node_rad = 0.3

    colors = {
        "Input\n(30 features)": "blue",
        "Hidden\n(64 neurons)": "green",
        "Output\n(1 neuron)": "red",
    }

    layer_nodes = {}

    for layer_name, (x, y_positions) in layers.items():
        layer_nodes[layer_name] = []
        for y in y_positions:
            circ = plt.Circle((x, y), node_rad, color=colors[layer_name], ec="black", zorder=3)
            axes.add_patch(circ)
            layer_nodes[layer_name].append((x, y))

        if layer_name == "Input\n(30 features)":
            axes.text(x, 1.2, "...", ha="center", fontsize=14, color="blue")
        if layer_name == "Hidden\n(64 neurons)":
            axes.text(x, 1.2, "...", ha="center", fontsize=14, color="green")

        axes.text(x, 0, layer_name, ha="center", fontsize=14, fontweight="bold")

    input_nodes = layer_nodes["Input\n(30 features)"]
    hidden_nodes = layer_nodes["Hidden\n(64 neurons)"]
    output_nodes = layer_nodes["Output\n(1 neuron)"]

    for (x1, y1) in input_nodes[::2]:
        for (x2, y2) in hidden_nodes:
            axes.plot([x1, x2], [y1, y2], "grey", alpha=0.5, linewidth=0.8, zorder=3)

    for (x1, y1) in hidden_nodes[::2]:
        for (x2, y2) in output_nodes:
            axes.plot([x1, x2], [y1, y2], "grey", alpha=0.5, linewidth=0.8, zorder=3)

    axes.text(3.2, 9.5, "ReLU", ha="center", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", edgecolor="black"))
    axes.text(6.8, 9.5, "Sigmoid", ha="center", fontsize=14, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", edgecolor="black"))

    axes.set_title("Neural Network Architecture (30 -> 64 -> 1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_class_dist(df, save_path="fig2_class_dist.png"):
    raw_count = df["Class"].value_counts().sort_index()
    fraud_count = int(raw_count[1])
    normal_count = int(raw_count[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].bar(["Normal", "Fraud"], [normal_count, fraud_count], color=["blue", "red"], edgecolor="black")
    axes[0].set_title("Before Undersampling")
    axes[0].set_ylabel("Transaction Count")
    for bar, val in zip(axes[0].patches, [normal_count, fraud_count]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500, f"{val:,}", ha="center", fontsize=10)

    axes[1].bar(["Normal", "Fraud"], [fraud_count, fraud_count], color=["blue", "red"], edgecolor="black")
    axes[1].set_title("After Undersampling")
    for bar in axes[1].patches:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{fraud_count:,}", ha="center",
                     fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_metric_comparison(results, save_path="fig3_metrics_compare.png"):
    labels = [r["label"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["F1"] for r in results]

    x = np.arange(len(labels))
    width = 0.1

    fig, axes = plt.subplots(figsize=(8, 6))
    axes.bar(x - 1.5 * width, accuracies, width, label="Accuracy", color="blue")
    axes.bar(x - 0.5 * width, precisions, width, label="Precision", color="red")
    axes.bar(x + 0.5 * width, recalls, width, label="Recall", color="green")
    axes.bar(x + 1.5 * width, f1s, width, label="F1", color="purple")

    axes.set_xticks(x)
    axes.set_xticklabels(labels, rotation=0)
    axes.set_ylim(0.4, 1)
    axes.set_yticks(np.arange(0.4, 1.025, 0.05))
    axes.set_ylabel("Score")
    axes.set_title("Performance Metrics Across Experiments")
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    axes.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_confusion_matrix(cm, title="Confusion Matrix", save_path="fig4_confusion_matrix.png"):
    cm = np.array(cm)
    fig, axes = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Fraud"],
        yticklabels=["Actual Normal", "Actual Fraud"],
        linewidths=1, linecolor="black", ax=axes,
    )
    axes.set_title(title)
    axes.set_ylabel("Actual Label")
    axes.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)