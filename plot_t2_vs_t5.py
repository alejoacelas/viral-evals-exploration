#!/usr/bin/env python3
"""Exploratory plots comparing 150M_T2 vs 150M_T5 perplexity differences."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

DATA_PATH = Path(__file__).parent / "combined_perplexity.parquet"
OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

EVAL_TIER = 4  # Filter to this evaluation tier only

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 9


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df[df["eval_tier"] == EVAL_TIER]
    print(f"Filtered to eval_tier={EVAL_TIER}: {len(df)} rows")
    return df


def pivot_for_comparison(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Create wide format with both models' perplexity for same sequences."""
    df_a = df[df["model"] == model_a][["sequence_id", "eval_tier", "perplexity"]].copy()
    df_b = df[df["model"] == model_b][["sequence_id", "eval_tier", "perplexity"]].copy()
    df_a = df_a.rename(columns={"perplexity": f"ppl_{model_a}"})
    df_b = df_b.rename(columns={"perplexity": f"ppl_{model_b}"})
    merged = df_a.merge(df_b, on=["sequence_id", "eval_tier"])
    merged["ppl_diff"] = merged[f"ppl_{model_a}"] - merged[f"ppl_{model_b}"]
    merged["ppl_ratio"] = merged[f"ppl_{model_a}"] / merged[f"ppl_{model_b}"]
    merged["log_ratio"] = np.log2(merged["ppl_ratio"])
    return merged


def plot_scatter_with_marginals(comp: pd.DataFrame, model_a: str, model_b: str):
    """Scatter plot of ppl_A vs ppl_B with marginal histograms."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[0.8, 3, 0.3], height_ratios=[0.8, 3, 0.3],
                          hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)

    ppl_a_col = f"ppl_{model_a}"
    ppl_b_col = f"ppl_{model_b}"

    # Color by difference magnitude
    colors = comp["ppl_diff"]
    sc = ax_main.scatter(comp[ppl_b_col], comp[ppl_a_col], alpha=0.5, s=20,
                         c=colors, cmap="RdYlBu_r", edgecolors="none")
    plt.colorbar(sc, ax=ax_main, label=f"Diff ({model_a} - {model_b})", shrink=0.6)

    # Diagonal line (equal perplexity)
    lims = [comp[[ppl_a_col, ppl_b_col]].min().min(), comp[[ppl_a_col, ppl_b_col]].max().max()]
    ax_main.plot(lims, lims, "k--", alpha=0.7, lw=1.5, label="y=x")
    ax_main.set_xlabel(f"{model_b} perplexity")
    ax_main.set_ylabel(f"{model_a} perplexity")

    # Marginal histograms
    ax_top.hist(comp[ppl_b_col], bins=50, alpha=0.7, color="steelblue", edgecolor="none")
    ax_right.hist(comp[ppl_a_col], bins=50, alpha=0.7, color="steelblue",
                  orientation="horizontal", edgecolor="none")

    ax_top.set_ylabel("Count")
    ax_right.set_xlabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Correlation stats
    r, p = stats.pearsonr(comp[ppl_a_col], comp[ppl_b_col])
    rho, _ = stats.spearmanr(comp[ppl_a_col], comp[ppl_b_col])
    ax_main.text(0.05, 0.95, f"Pearson r={r:.3f}\nSpearman ρ={rho:.3f}\nN={len(comp)}",
                 transform=ax_main.transAxes, va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax_main.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{model_a} vs {model_b}: Per-sequence Perplexity (Tier {EVAL_TIER} only)", y=0.98)
    plt.savefig(OUT_DIR / f"scatter_{model_a}_vs_{model_b}_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_difference_distribution(comp: pd.DataFrame, model_a: str, model_b: str):
    """Distribution of perplexity differences."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram of differences
    ax = axes[0]
    n, bins, patches = ax.hist(comp["ppl_diff"], bins=60, alpha=0.7, edgecolor="black", linewidth=0.3)
    # Color bars by sign
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor("steelblue")
        else:
            patch.set_facecolor("coral")
    ax.axvline(0, color="black", linestyle="-", lw=2)
    ax.axvline(comp["ppl_diff"].mean(), color="green", linestyle="--", lw=2, label=f"mean={comp['ppl_diff'].mean():.2f}")
    ax.axvline(comp["ppl_diff"].median(), color="purple", linestyle=":", lw=2, label=f"median={comp['ppl_diff'].median():.2f}")
    ax.set_xlabel(f"Perplexity diff ({model_a} - {model_b})")
    ax.set_ylabel("Count")
    ax.set_title(f"Difference Distribution (N={len(comp)})")
    ax.legend(fontsize=8)
    ax.text(0.95, 0.95, f">0: {(comp['ppl_diff']>0).sum()} ({(comp['ppl_diff']>0).mean()*100:.1f}%)\n<0: {(comp['ppl_diff']<0).sum()} ({(comp['ppl_diff']<0).mean()*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # Log ratio distribution
    ax = axes[1]
    ax.hist(comp["log_ratio"], bins=60, alpha=0.7, edgecolor="black", linewidth=0.3, color="teal")
    ax.axvline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel(f"log2({model_a} / {model_b})")
    ax.set_ylabel("Count")
    ax.set_title("Log Ratio Distribution")

    # QQ plot to check normality
    ax = axes[2]
    stats.probplot(comp["ppl_diff"], dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (vs Normal)")

    plt.suptitle(f"Perplexity Difference Analysis: {model_a} - {model_b} (Tier {EVAL_TIER})", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"diff_dist_{model_a}_vs_{model_b}_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ranked_differences(comp: pd.DataFrame, model_a: str, model_b: str):
    """Waterfall-style plot showing sorted differences - reveals if changes are concentrated."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by difference
    sorted_diff = comp.sort_values("ppl_diff")
    x = np.arange(len(sorted_diff))

    ax = axes[0]
    colors = ["steelblue" if d < 0 else "coral" for d in sorted_diff["ppl_diff"]]
    ax.bar(x, sorted_diff["ppl_diff"], color=colors, width=1.0, alpha=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Sequence (sorted by difference)")
    ax.set_ylabel(f"Perplexity diff ({model_a} - {model_b})")
    ax.set_title("Sorted Differences: Is change concentrated or distributed?")

    # Cumulative contribution
    ax = axes[1]
    sorted_abs = comp.assign(abs_diff=comp["ppl_diff"].abs()).sort_values("abs_diff", ascending=False)
    cumsum = sorted_abs["abs_diff"].cumsum()
    total = cumsum.iloc[-1]
    ax.plot(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum / total * 100, lw=2, color="teal")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Uniform distribution")
    ax.set_xlabel("% of sequences (sorted by |diff|)")
    ax.set_ylabel("% of cumulative absolute difference")
    ax.set_title("Concentration curve: How many sequences drive the difference?")
    ax.fill_between(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum / total * 100,
                    np.arange(len(cumsum)) / len(cumsum) * 100, alpha=0.2, color="teal")

    # Mark key percentiles
    for pct in [10, 20, 50]:
        idx = int(len(cumsum) * pct / 100)
        contrib = cumsum.iloc[idx] / total * 100
        ax.scatter([pct], [contrib], s=80, zorder=5, color="darkred")
        ax.annotate(f"Top {pct}% → {contrib:.0f}%", (pct, contrib), textcoords="offset points",
                    xytext=(8, -5), fontsize=9, fontweight="bold")

    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.suptitle(f"Concentration Analysis (Tier {EVAL_TIER}): Are differences driven by few sequences?", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ranked_diff_{model_a}_vs_{model_b}_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_extreme_sequences(comp: pd.DataFrame, model_a: str, model_b: str, n=30):
    """Show sequences with largest positive/negative changes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    ppl_a_col = f"ppl_{model_a}"
    ppl_b_col = f"ppl_{model_b}"

    # Top increases (T2 > T5)
    top_increase = comp.nlargest(n, "ppl_diff")
    ax = axes[0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, n))
    bars = ax.barh(range(n), top_increase["ppl_diff"].values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{s}" for s in top_increase["sequence_id"]], fontsize=7)
    ax.set_xlabel(f"Perplexity diff ({model_a} - {model_b})")
    ax.set_title(f"Top {n} sequences where {model_a} >> {model_b}\n(higher ppl in T2)")
    ax.invert_yaxis()
    # Add actual values as text
    for i, (idx, row) in enumerate(top_increase.iterrows()):
        ax.text(row["ppl_diff"] + 0.1, i, f"{row[ppl_a_col]:.1f} vs {row[ppl_b_col]:.1f}",
                va="center", fontsize=6)

    # Top decreases (T2 < T5)
    top_decrease = comp.nsmallest(n, "ppl_diff")
    ax = axes[1]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n))
    bars = ax.barh(range(n), top_decrease["ppl_diff"].values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{s}" for s in top_decrease["sequence_id"]], fontsize=7)
    ax.set_xlabel(f"Perplexity diff ({model_a} - {model_b})")
    ax.set_title(f"Top {n} sequences where {model_a} << {model_b}\n(lower ppl in T2)")
    ax.invert_yaxis()
    # Add actual values as text
    for i, (idx, row) in enumerate(top_decrease.iterrows()):
        ax.text(row["ppl_diff"] - 0.1, i, f"{row[ppl_a_col]:.1f} vs {row[ppl_b_col]:.1f}",
                va="center", ha="right", fontsize=6)

    plt.suptitle(f"Extreme Outliers: {model_a} vs {model_b} (Tier {EVAL_TIER})", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"extreme_seqs_{model_a}_vs_{model_b}_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, param_size: str = "150M"):
    """Correlation matrix across all models of a given size."""
    models = sorted([m for m in df["model"].unique() if m.startswith(param_size)])

    # Create wide format: one column per model
    pivoted = df[df["model"].isin(models)].pivot_table(
        index="sequence_id", columns="model", values="perplexity"
    ).reset_index()

    # Compute correlation matrix
    corr_matrix = pivoted[models].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.9, vmax=1.0,
                mask=mask, square=True, ax=ax, annot_kws={"size": 10})
    ax.set_title(f"Perplexity Correlation Matrix ({param_size} models, Tier {EVAL_TIER})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"corr_matrix_{param_size}_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return corr_matrix


def plot_pairwise_scatter_grid(df: pd.DataFrame, focus_models: list[str], sample_n=2000):
    """Grid of pairwise scatter plots between specified models."""
    n = len(focus_models)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))

    # Pivot to wide format
    pivoted = df[df["model"].isin(focus_models)].pivot_table(
        index="sequence_id", columns="model", values="perplexity"
    ).reset_index()

    if len(pivoted) > sample_n:
        pivoted = pivoted.sample(sample_n, random_state=42)

    for i, m1 in enumerate(focus_models):
        for j, m2 in enumerate(focus_models):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram
                ax.hist(pivoted[m1], bins=40, alpha=0.7, edgecolor="none", color="steelblue")
                ax.set_ylabel("Count" if j == 0 else "")
            elif i > j:
                # Lower triangle: scatter
                ax.scatter(pivoted[m2], pivoted[m1], alpha=0.4, s=8, edgecolors="none")
                r, _ = stats.pearsonr(pivoted[m1], pivoted[m2])
                ax.text(0.05, 0.95, f"r={r:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
                lims = [min(pivoted[m1].min(), pivoted[m2].min()),
                        max(pivoted[m1].max(), pivoted[m2].max())]
                ax.plot(lims, lims, "r--", alpha=0.5, lw=1)
            else:
                # Upper triangle: empty or annotated
                ax.axis("off")

            if i == n-1:
                ax.set_xlabel(m2.split("_")[1] if "_" in m2 else m2, fontsize=9)
            if j == 0 and i != j:
                ax.set_ylabel(m1.split("_")[1] if "_" in m1 else m1, fontsize=9)

    plt.suptitle(f"Pairwise Perplexity Comparisons (150M models, Tier {EVAL_TIER})", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"pairwise_scatter_150M_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_diff_by_base_perplexity(comp: pd.DataFrame, model_a: str, model_b: str):
    """Is the difference related to baseline perplexity?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ppl_b_col = f"ppl_{model_b}"
    ppl_a_col = f"ppl_{model_a}"

    # Scatter: base perplexity vs difference
    ax = axes[0]
    sc = ax.scatter(comp[ppl_b_col], comp["ppl_diff"], alpha=0.5, s=15,
                    c=comp[ppl_a_col], cmap="viridis", edgecolors="none")
    plt.colorbar(sc, ax=ax, label=f"{model_a} ppl")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel(f"Baseline perplexity ({model_b})")
    ax.set_ylabel(f"Perplexity diff ({model_a} - {model_b})")
    ax.set_title("Difference vs Baseline Perplexity")

    # Binned analysis
    ax = axes[1]
    comp_copy = comp.copy()
    comp_copy["ppl_bin"] = pd.cut(comp_copy[ppl_b_col], bins=10)
    bin_stats = comp_copy.groupby("ppl_bin", observed=True)["ppl_diff"].agg(["mean", "std", "count"])
    bin_centers = [interval.mid for interval in bin_stats.index]
    ax.errorbar(bin_centers, bin_stats["mean"], yerr=bin_stats["std"], fmt="o-", capsize=4,
                color="teal", markersize=8, linewidth=2)
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel(f"Baseline perplexity bin ({model_b})")
    ax.set_ylabel(f"Mean diff ± std")
    ax.set_title("Binned: Mean difference by baseline perplexity")
    ax.tick_params(axis="x", rotation=45)
    # Add count annotations
    for x, row in zip(bin_centers, bin_stats.itertuples()):
        ax.annotate(f"n={row.count}", (x, row.mean), textcoords="offset points",
                    xytext=(0, 10), fontsize=7, ha="center")

    # Hexbin for density
    ax = axes[2]
    hb = ax.hexbin(comp[ppl_b_col], comp["ppl_diff"], gridsize=25, cmap="YlOrRd", mincnt=1)
    ax.axhline(0, color="black", linestyle="--", lw=1.5)
    ax.set_xlabel(f"Baseline perplexity ({model_b})")
    ax.set_ylabel(f"Perplexity diff ({model_a} - {model_b})")
    ax.set_title("Density: Where are most sequences?")
    plt.colorbar(hb, ax=ax, label="Count")

    plt.suptitle(f"Relationship between baseline perplexity and model difference (Tier {EVAL_TIER})", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"diff_vs_baseline_{model_a}_vs_{model_b}_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_dashboard(df: pd.DataFrame, comp: pd.DataFrame, model_a: str, model_b: str):
    """Multi-panel summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ppl_a_col = f"ppl_{model_a}"
    ppl_b_col = f"ppl_{model_b}"

    # Panel 1: Scatter with density
    ax = fig.add_subplot(gs[0, 0])
    ax.hexbin(comp[ppl_b_col], comp[ppl_a_col], gridsize=30, cmap="Blues", mincnt=1)
    lims = [comp[[ppl_a_col, ppl_b_col]].min().min(), comp[[ppl_a_col, ppl_b_col]].max().max()]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel(f"{model_b} ppl")
    ax.set_ylabel(f"{model_a} ppl")
    ax.set_title("A. Pairwise perplexity")

    # Panel 2: Difference histogram
    ax = fig.add_subplot(gs[0, 1])
    n, bins, patches = ax.hist(comp["ppl_diff"], bins=50, alpha=0.7, edgecolor="none")
    for i, patch in enumerate(patches):
        patch.set_facecolor("steelblue" if bins[i] < 0 else "coral")
    ax.axvline(0, color="black", linestyle="-", lw=2)
    ax.axvline(comp["ppl_diff"].mean(), color="green", linestyle="--", lw=2)
    ax.set_xlabel("Ppl diff (T2 - T5)")
    ax.set_ylabel("Count")
    ax.set_title(f"B. Diff distribution (μ={comp['ppl_diff'].mean():.2f})")

    # Panel 3: Cumulative concentration
    ax = fig.add_subplot(gs[0, 2])
    sorted_abs = comp.assign(abs_diff=comp["ppl_diff"].abs()).sort_values("abs_diff", ascending=False)
    cumsum = sorted_abs["abs_diff"].cumsum() / sorted_abs["abs_diff"].sum() * 100
    ax.plot(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum, lw=2, color="teal")
    ax.fill_between(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum,
                    np.arange(len(cumsum)) / len(cumsum) * 100, alpha=0.2, color="teal")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5)
    ax.set_xlabel("% sequences")
    ax.set_ylabel("% cumulative |diff|")
    ax.set_title("C. Concentration curve")

    # Panel 4: Perplexity comparison across all 150M models
    ax = fig.add_subplot(gs[1, 0])
    models_150m = sorted([m for m in df["model"].unique() if m.startswith("150M")])
    model_means = df[df["model"].isin(models_150m)].groupby("model")["perplexity"].mean()
    colors = ["coral" if m in [model_a, model_b] else "steelblue" for m in model_means.index]
    model_means.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean perplexity")
    ax.set_title("D. Mean ppl across 150M models")
    ax.tick_params(axis="x", rotation=45)

    # Panel 5: Diff vs baseline
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(comp[ppl_b_col], comp["ppl_diff"], alpha=0.4, s=10,
                    c=comp[ppl_a_col], cmap="viridis", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel(f"Baseline ppl ({model_b})")
    ax.set_ylabel("Ppl diff")
    ax.set_title("E. Diff vs baseline")
    plt.colorbar(sc, ax=ax, label=f"{model_a} ppl", shrink=0.8)

    # Panel 6: Sorted differences waterfall
    ax = fig.add_subplot(gs[1, 2])
    sorted_diff = comp.sort_values("ppl_diff")
    colors = ["steelblue" if d < 0 else "coral" for d in sorted_diff["ppl_diff"]]
    ax.bar(range(len(sorted_diff)), sorted_diff["ppl_diff"], color=colors, width=1.0, alpha=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Sequence (sorted)")
    ax.set_ylabel("Ppl diff")
    ax.set_title("F. Sorted differences")

    # Panel 7: Log ratio histogram
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(comp["log_ratio"], bins=50, alpha=0.7, edgecolor="none", color="teal")
    ax.axvline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("log2(T2/T5)")
    ax.set_ylabel("Count")
    ax.set_title("G. Log ratio distribution")

    # Panel 8: Stats summary
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    r, _ = stats.pearsonr(comp[ppl_a_col], comp[ppl_b_col])
    rho, _ = stats.spearmanr(comp[ppl_a_col], comp[ppl_b_col])
    stats_text = f"""
    Summary Statistics (Tier {EVAL_TIER})
    ─────────────────────────────
    Total sequences: {len(comp):,}

    Pearson r: {r:.4f}
    Spearman ρ: {rho:.4f}

    Mean diff (T2-T5): {comp['ppl_diff'].mean():.3f}
    Median diff: {comp['ppl_diff'].median():.3f}
    Std diff: {comp['ppl_diff'].std():.3f}

    T2 > T5: {(comp['ppl_diff'] > 0).sum():,} ({(comp['ppl_diff'] > 0).mean()*100:.1f}%)
    T2 < T5: {(comp['ppl_diff'] < 0).sum():,} ({(comp['ppl_diff'] < 0).mean()*100:.1f}%)

    Top 10% contribute: {sorted_abs['abs_diff'].head(int(len(sorted_abs)*0.1)).sum() / sorted_abs['abs_diff'].sum() * 100:.1f}% of |diff|
    """
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10, family="monospace",
            va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Panel 9: Extreme sequences
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    top5_up = comp.nlargest(5, "ppl_diff")[["sequence_id", ppl_a_col, ppl_b_col, "ppl_diff"]]
    top5_down = comp.nsmallest(5, "ppl_diff")[["sequence_id", ppl_a_col, ppl_b_col, "ppl_diff"]]
    text = "Top 5 T2 >> T5:\n"
    for _, row in top5_up.iterrows():
        text += f"  {row['sequence_id'][:18]}: +{row['ppl_diff']:.2f}\n"
    text += "\nTop 5 T2 << T5:\n"
    for _, row in top5_down.iterrows():
        text += f"  {row['sequence_id'][:18]}: {row['ppl_diff']:.2f}\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8, family="monospace",
            va="top", bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))

    fig.suptitle(f"150M_T2 vs 150M_T5 Perplexity Comparison Dashboard (Tier {EVAL_TIER})", fontsize=14, y=0.98)
    plt.savefig(OUT_DIR / f"dashboard_T2_vs_T5_tier{EVAL_TIER}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading data...")
    df = load_data()

    model_a, model_b = "150M_T2", "150M_T5"
    print(f"Comparing {model_a} vs {model_b}...")
    comp = pivot_for_comparison(df, model_a, model_b)

    print("Generating plots...")

    # Main comparison plots
    plot_scatter_with_marginals(comp, model_a, model_b)
    print("  - Scatter with marginals")

    plot_difference_distribution(comp, model_a, model_b)
    print("  - Difference distribution")

    plot_ranked_differences(comp, model_a, model_b)
    print("  - Ranked differences (concentration)")

    plot_extreme_sequences(comp, model_a, model_b)
    print("  - Extreme sequences")

    plot_diff_by_base_perplexity(comp, model_a, model_b)
    print("  - Diff vs baseline perplexity")

    # Correlation analyses
    plot_correlation_matrix(df, "150M")
    print("  - Correlation matrix (150M)")

    focus_models = ["150M_F", "150M_H", "150M_T1", "150M_T2", "150M_T5", "150M_T6"]
    plot_pairwise_scatter_grid(df, focus_models)
    print("  - Pairwise scatter grid")

    # Summary dashboard
    plot_summary_dashboard(df, comp, model_a, model_b)
    print("  - Summary dashboard")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
