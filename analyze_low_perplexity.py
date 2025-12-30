"""
Analyze what characterizes low-perplexity sequences in Tier 4.

Compares sequences with perplexity < 10 (low mode) vs >= 15 (high mode)
to understand the bimodal distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def load_data():
    """Load perplexity data and metadata, merge them."""
    # Load perplexity (use 150M_F model for Tier 4)
    perp = pd.read_csv('combined_perplexity.csv')
    tier4 = perp[(perp['eval_tier'] == 4) & (perp['model'] == '150M_F')].copy()

    # Load metadata
    meta = pd.read_csv('uniref_metadata.csv')

    # Merge
    df = tier4.merge(meta, on='sequence_id', how='left')

    # Add perplexity group
    df['perp_group'] = pd.cut(
        df['perplexity'],
        bins=[0, 10, 15, 100],
        labels=['low (<10)', 'moderate (10-15)', 'high (>=15)']
    )

    print(f"Loaded {len(df)} sequences")
    print(f"\nPerplexity groups:")
    print(df['perp_group'].value_counts().sort_index())

    return df


def statistical_comparison(df, feature_cols):
    """
    Perform statistical tests comparing low vs high perplexity groups.
    """
    low = df[df['perp_group'] == 'low (<10)']
    high = df[df['perp_group'] == 'high (>=15)']

    results = []
    for col in feature_cols:
        if col not in df.columns:
            continue

        low_vals = low[col].dropna()
        high_vals = high[col].dropna()

        if len(low_vals) < 5 or len(high_vals) < 5:
            continue

        # Mann-Whitney U test
        stat, pval = stats.mannwhitneyu(low_vals, high_vals, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(low_vals), len(high_vals)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        results.append({
            'feature': col,
            'low_mean': low_vals.mean(),
            'low_median': low_vals.median(),
            'high_mean': high_vals.mean(),
            'high_median': high_vals.median(),
            'mannwhitney_stat': stat,
            'p_value': pval,
            'effect_size': effect_size
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')

    return results_df


def plot_perplexity_vs_length(df, output_dir):
    """Scatter plot of perplexity vs sequence length."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df['seq_length'],
        df['perplexity'],
        c=df['perplexity'],
        cmap='viridis',
        alpha=0.6,
        s=30
    )

    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Tier 4: Perplexity vs Sequence Length (150M_F)', fontsize=14)

    plt.colorbar(scatter, ax=ax, label='Perplexity')

    # Add correlation
    valid = df[['seq_length', 'perplexity']].dropna()
    r, p = stats.pearsonr(valid['seq_length'], valid['perplexity'])
    ax.text(
        0.05, 0.95,
        f'r = {r:.3f}, p = {p:.2e}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity_vs_length.png', dpi=150)
    plt.close()
    print(f"  Saved perplexity_vs_length.png (r={r:.3f})")


def plot_feature_distributions(df, features, output_dir):
    """Plot distributions of key features by perplexity group."""
    # Filter to low and high groups only
    df_filtered = df[df['perp_group'].isin(['low (<10)', 'high (>=15)'])].copy()

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        if feat not in df_filtered.columns:
            axes[i].set_visible(False)
            continue

        ax = axes[i]
        sns.boxplot(data=df_filtered, x='perp_group', y=feat, ax=ax, palette='Set2')
        ax.set_xlabel('')
        ax.set_ylabel(feat, fontsize=10)
        ax.set_title(feat, fontsize=11)

    # Hide unused axes
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distributions: Low vs High Perplexity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_distributions.png")


def plot_taxonomy_comparison(df, output_dir):
    """Compare taxonomy between low and high perplexity groups."""
    low = df[df['perp_group'] == 'low (<10)']
    high = df[df['perp_group'] == 'high (>=15)']

    # Get top taxa
    all_taxa = df['common_taxon'].value_counts().head(15).index.tolist()

    # Count by group
    low_counts = low['common_taxon'].value_counts()
    high_counts = high['common_taxon'].value_counts()

    tax_data = []
    for taxon in all_taxa:
        tax_data.append({
            'taxon': taxon[:40] + '...' if len(taxon) > 40 else taxon,
            'low_count': low_counts.get(taxon, 0),
            'high_count': high_counts.get(taxon, 0)
        })

    tax_df = pd.DataFrame(tax_data)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(tax_df))
    width = 0.35

    bars1 = ax.barh(x - width/2, tax_df['low_count'], width, label='Low (<10)', color='steelblue')
    bars2 = ax.barh(x + width/2, tax_df['high_count'], width, label='High (>=15)', color='coral')

    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Common Taxon', fontsize=12)
    ax.set_title('Taxonomy Distribution by Perplexity Group', fontsize=14)
    ax.set_yticks(x)
    ax.set_yticklabels(tax_df['taxon'])
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'taxonomy_comparison.png', dpi=150)
    plt.close()
    print("  Saved taxonomy_comparison.png")


def plot_complexity_scatter(df, output_dir):
    """Scatter plot of complexity metrics colored by perplexity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Local complexity vs perplexity
    ax = axes[0]
    scatter = ax.scatter(
        df['local_complexity'],
        df['perplexity'],
        c=df['seq_length'],
        cmap='plasma',
        alpha=0.6,
        s=30
    )
    ax.set_xlabel('Local Complexity (window entropy)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Local Complexity vs Perplexity')
    plt.colorbar(scatter, ax=ax, label='Length')

    # 2. Repeat fraction vs perplexity
    ax = axes[1]
    scatter = ax.scatter(
        df['repeat_fraction'],
        df['perplexity'],
        c=df['seq_length'],
        cmap='plasma',
        alpha=0.6,
        s=30
    )
    ax.set_xlabel('Repeat Fraction')
    ax.set_ylabel('Perplexity')
    ax.set_title('Repeat Fraction vs Perplexity')
    plt.colorbar(scatter, ax=ax, label='Length')

    # 3. Shannon entropy vs perplexity
    ax = axes[2]
    scatter = ax.scatter(
        df['shannon_entropy'],
        df['perplexity'],
        c=df['seq_length'],
        cmap='plasma',
        alpha=0.6,
        s=30
    )
    ax.set_xlabel('Shannon Entropy')
    ax.set_ylabel('Perplexity')
    ax.set_title('Shannon Entropy vs Perplexity')
    plt.colorbar(scatter, ax=ax, label='Length')

    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_vs_perplexity.png', dpi=150)
    plt.close()
    print("  Saved complexity_vs_perplexity.png")


def plot_length_histogram_by_group(df, output_dir):
    """Histogram of sequence lengths split by perplexity group."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_filtered = df[df['perp_group'].isin(['low (<10)', 'high (>=15)'])]

    for group, color in [('low (<10)', 'steelblue'), ('high (>=15)', 'coral')]:
        group_data = df_filtered[df_filtered['perp_group'] == group]['seq_length']
        ax.hist(
            group_data,
            bins=30,
            alpha=0.6,
            label=f'{group} (n={len(group_data)}, median={group_data.median():.0f})',
            color=color
        )

    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Sequence Length Distribution by Perplexity Group', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'length_by_group.png', dpi=150)
    plt.close()
    print("  Saved length_by_group.png")


def print_extreme_examples(df):
    """Print examples of extremely low and high perplexity sequences."""
    print("\n" + "="*70)
    print("EXTREME EXAMPLES")
    print("="*70)

    # Lowest perplexity
    print("\n--- 10 Lowest Perplexity Sequences ---")
    lowest = df.nsmallest(10, 'perplexity')[
        ['sequence_id', 'perplexity', 'seq_length', 'common_taxon', 'name']
    ]
    for _, row in lowest.iterrows():
        name_short = (row['name'][:50] + '...') if len(str(row['name'])) > 50 else row['name']
        print(f"  {row['perplexity']:.2f} | len={row['seq_length']:4d} | {row['common_taxon'][:30]}")
        print(f"         {name_short}")

    # Highest perplexity
    print("\n--- 10 Highest Perplexity Sequences ---")
    highest = df.nlargest(10, 'perplexity')[
        ['sequence_id', 'perplexity', 'seq_length', 'common_taxon', 'name']
    ]
    for _, row in highest.iterrows():
        name_short = (row['name'][:50] + '...') if len(str(row['name'])) > 50 else row['name']
        print(f"  {row['perplexity']:.2f} | len={row['seq_length']:4d} | {row['common_taxon'][:30]}")
        print(f"         {name_short}")


def main():
    # Setup output directory
    output_dir = Path('plots/low_perplexity_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    # Key features to analyze
    feature_cols = [
        'seq_length',
        'shannon_entropy',
        'local_complexity',
        'max_homopolymer',
        'repeat_fraction',
        'hydrophobicity_mean',
        'net_charge',
        'fraction_charged',
        'member_count',
    ]

    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (Low <10 vs High >=15)")
    print("="*70)
    stat_results = statistical_comparison(df, feature_cols)
    print("\nTop features by significance:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(stat_results.to_string(index=False))

    # Save statistical results
    stat_results.to_csv(output_dir / 'statistical_comparison.csv', index=False)
    print(f"\nSaved statistical_comparison.csv")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_perplexity_vs_length(df, output_dir)
    plot_length_histogram_by_group(df, output_dir)
    plot_feature_distributions(df, feature_cols, output_dir)
    plot_taxonomy_comparison(df, output_dir)
    plot_complexity_scatter(df, output_dir)

    # Print extreme examples
    print_extreme_examples(df)

    # Summary
    print("\n" + "="*70)
    print("KEY FINDINGS SUMMARY")
    print("="*70)

    low = df[df['perp_group'] == 'low (<10)']
    high = df[df['perp_group'] == 'high (>=15)']

    print(f"\n1. SEQUENCE LENGTH:")
    print(f"   Low perplexity:  median = {low['seq_length'].median():.0f}, mean = {low['seq_length'].mean():.1f}")
    print(f"   High perplexity: median = {high['seq_length'].median():.0f}, mean = {high['seq_length'].mean():.1f}")

    # Correlation
    r, p = stats.pearsonr(df['seq_length'].dropna(), df['perplexity'].dropna())
    print(f"   Correlation: r = {r:.3f} (p = {p:.2e})")

    print(f"\n2. SEQUENCE COMPLEXITY:")
    print(f"   Local complexity - Low: {low['local_complexity'].mean():.3f}, High: {high['local_complexity'].mean():.3f}")
    print(f"   Repeat fraction  - Low: {low['repeat_fraction'].mean():.4f}, High: {high['repeat_fraction'].mean():.4f}")

    print(f"\n3. TAXONOMY:")
    print(f"   Low perplexity - top taxa:")
    for taxon, count in low['common_taxon'].value_counts().head(5).items():
        print(f"      {taxon}: {count} ({100*count/len(low):.1f}%)")
    print(f"   High perplexity - top taxa:")
    for taxon, count in high['common_taxon'].value_counts().head(5).items():
        print(f"      {taxon}: {count} ({100*count/len(high):.1f}%)")

    print(f"\nOutput saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
