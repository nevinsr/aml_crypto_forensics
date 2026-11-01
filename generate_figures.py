"""
Academic Figure Generation Script
=================================
Generates publication-ready, high-resolution (300 DPI) figures for:
- Figure 2: Side-by-side Confusion Matrices (LightGBM vs. GCN)
- Figure 3: Comparative Performance Bar Charts (Percentages & Decimals)

Designed for academic research papers (IEEE / ACM / Springer style).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


def set_academic_style():
    """Configures global matplotlib and seaborn rcParams for publication quality."""
    sns.set_theme(style="white", font="DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.2,
        "axes.labelweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11.5,
        "figure.titlesize": 14.5,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def generate_figure_2(save_path: str = "fig2_confusion_matrices.png"):
    """
    Figure 2: Side-by-side Confusion Matrices (1x2 subplot)
    - Left Matrix (LightGBM): TP=801, FP=191, FN=282, TN=15396
    - Right Matrix (GCN): TP=581, FP=790, FN=502, TN=14797
    - Labels: Y-axis 'Actual Class' (Illicit, Licit), X-axis 'Predicted Class' (Illicit, Licit)
    """
    print(f"Generating Figure 2: {save_path} ...")
    
    # Confusion Matrix arrays
    # Row 0: Actual Illicit -> [TP, FN] (Predicted Illicit, Predicted Licit)
    # Row 1: Actual Licit   -> [FP, TN] (Predicted Illicit, Predicted Licit)
    cm_lgbm = np.array([[801, 282],
                        [191, 15396]])
    
    cm_gcn = np.array([[581, 502],
                       [790, 14797]])
    
    labels = ["Illicit", "Licit"]
    vmax = max(cm_lgbm.max(), cm_gcn.max())
    
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    
    models = [
        {"name": "(a) LightGBM Baseline", "data": cm_lgbm, "ax": axes[0], "show_ylabel": True},
        {"name": "(b) GCN Model", "data": cm_gcn, "ax": axes[1], "show_ylabel": False}
    ]
    
    mesh = None
    for item in models:
        ax = item["ax"]
        data = item["data"]
        
        # Format raw numbers with comma thousands separator
        annot_labels = np.array([[f"{val:,}" for val in row] for row in data])
        
        mesh = sns.heatmap(
            data,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 13.5, "weight": "bold"},
            linewidths=1.5,
            linecolor="#E2E8F0"
        )
        
        ax.set_title(item["name"], pad=14, fontsize=13.5, weight="bold", color="#0F172A")
        ax.set_xlabel("Predicted Class", labelpad=10, fontsize=12, weight="bold", color="#0F172A")
        
        if item["show_ylabel"]:
            ax.set_ylabel("Actual Class", labelpad=10, fontsize=12, weight="bold", color="#0F172A")
        else:
            ax.set_ylabel("")
        
        ax.tick_params(axis="both", which="major", labelsize=11, length=0)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_yticklabels(labels, rotation=0, va="center")

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.18, 0.022, 0.68])
    cbar = fig.colorbar(mesh.get_children()[0], cax=cbar_ax)
    cbar.set_label("Number of Samples", fontsize=11.5, weight="bold", labelpad=10, color="#0F172A")
    cbar.ax.tick_params(labelsize=10.5)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    
    plt.subplots_adjust(left=0.09, right=0.90, wspace=0.22, top=0.88, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {os.path.abspath(save_path)}")


def generate_figure_3(save_path: str = "fig3_performance_metrics.png"):
    """
    Figure 3: Comparative Performance Bar Charts (1x2 subplot)
    - Subplot 1 (Percentages): Precision & Recall (LightGBM vs. GCN), Y-axis 0 to 100
    - Subplot 2 (Decimals): F1-Score & PR-AUC (LightGBM vs. GCN), Y-axis 0 to 1.0
    - Includes data labels on bars and visible legend.
    """
    print(f"Generating Figure 3: {save_path} ...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    
    # Academic Colors: Deep Navy (LightGBM) & Academic Terracotta (GCN)
    c_lgbm = "#1F4E79"
    c_gcn = "#D95F02"
    
    bar_width = 0.28
    
    # -------------------------------------------------------------
    # Subplot 1: Percentage Metrics (Precision & Recall)
    # -------------------------------------------------------------
    ax1 = axes[0]
    metrics_pct = ["Precision", "Recall"]
    lgbm_pct = [80.75, 73.96]
    gcn_pct = [42.38, 53.65]
    
    x1 = np.arange(len(metrics_pct))
    
    rects1_lgbm = ax1.bar(
        x1 - bar_width / 2, lgbm_pct, bar_width,
        label="LightGBM", color=c_lgbm, edgecolor="#0F2438", linewidth=1.1, zorder=3
    )
    rects1_gcn = ax1.bar(
        x1 + bar_width / 2, gcn_pct, bar_width,
        label="GCN", color=c_gcn, edgecolor="#873600", linewidth=1.1, zorder=3
    )
    
    ax1.set_title("(a) Precision & Recall (%)", pad=14, fontsize=13.5, weight="bold", color="#0F172A")
    ax1.set_ylabel("Score (%)", fontsize=12, weight="bold", color="#0F172A")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metrics_pct, fontsize=11.5, weight="bold")
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax1.grid(axis="y", linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    
    # Data labels on bars
    for rect in rects1_lgbm:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, weight="bold", color="#0F2438"
        )
    for rect in rects1_gcn:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, weight="bold", color="#873600"
        )
    
    sns.despine(ax=ax1, top=True, right=True)

    # -------------------------------------------------------------
    # Subplot 2: Decimal Metrics (F1-Score & PR-AUC)
    # -------------------------------------------------------------
    ax2 = axes[1]
    metrics_dec = ["F1-Score", "PR-AUC"]
    lgbm_dec = [0.77, 0.80]
    gcn_dec = [0.47, 0.48]
    
    x2 = np.arange(len(metrics_dec))
    
    rects2_lgbm = ax2.bar(
        x2 - bar_width / 2, lgbm_dec, bar_width,
        label="LightGBM", color=c_lgbm, edgecolor="#0F2438", linewidth=1.1, zorder=3
    )
    rects2_gcn = ax2.bar(
        x2 + bar_width / 2, gcn_dec, bar_width,
        label="GCN", color=c_gcn, edgecolor="#873600", linewidth=1.1, zorder=3
    )
    
    ax2.set_title("(b) F1-Score & PR-AUC", pad=14, fontsize=13.5, weight="bold", color="#0F172A")
    ax2.set_ylabel("Score (0.0 – 1.0)", fontsize=12, weight="bold", color="#0F172A")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_dec, fontsize=11.5, weight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax2.grid(axis="y", linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    
    # Data labels on bars
    for rect in rects2_lgbm:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, weight="bold", color="#0F2438"
        )
    for rect in rects2_gcn:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, weight="bold", color="#873600"
        )
    
    sns.despine(ax=ax2, top=True, right=True)

    # Add a clean, unified top-center legend
    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels_leg,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        framealpha=0.95,
        fontsize=11.5
    )

    plt.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.98, wspace=0.22)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {os.path.abspath(save_path)}")


def main():
    set_academic_style()
    generate_figure_2("fig2_confusion_matrices.png")
    generate_figure_3("fig3_performance_metrics.png")
    print(" Both figures successfully generated at 300 DPI!")


if __name__ == "__main__":
    main()
