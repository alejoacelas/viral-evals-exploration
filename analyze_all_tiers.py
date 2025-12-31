"""
Analyze what characterizes low-perplexity sequences across all tiers.

Compares sequences with perplexity < 10 (low mode) vs >= 15 (high mode)
to understand the bimodal distribution.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def load_data():
    """Load perplexity data and metadata, merge them."""
    # Load perplexity (use 150M_F model)
    perp = pd.read_csv('combined_perplexity.csv')
    perp_f = perp[perp['model'] == '150M_F'].copy()

    # Load metadata
    meta = pd.read_csv('uniref_metadata.csv')

    # Merge
    df = perp_f.merge(meta, on='sequence_id', how='left')

    # Add perplexity group
    df['perp_group'] = pd.cut(
        df['perplexity'],
        bins=[0, 10, 15, 100],
        labels=['low (<10)', 'moderate (10-15)', 'high (>=15)']
    )

    print(f"Loaded {len(df)} sequence-tier combinations")
    print(f"Tiers: {sorted(df['eval_tier'].unique())}")

    return df


def statistical_comparison(df, feature_cols):
    """Perform statistical tests comparing low vs high perplexity groups."""
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

        stat, pval = stats.mannwhitneyu(low_vals, high_vals, alternative='two-sided')
        n1, n2 = len(low_vals), len(high_vals)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        results.append({
            'feature': col,
            'low_mean': low_vals.mean(),
            'low_median': low_vals.median(),
            'high_mean': high_vals.mean(),
            'high_median': high_vals.median(),
            'p_value': pval,
            'effect_size': effect_size
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    return results_df


def plot_combined_taxonomy(df, output_dir):
    """Create a single figure with taxonomy comparison for all tiers."""
    tiers = sorted(df['eval_tier'].unique())
    n_tiers = len(tiers)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        tier_df = df[df['eval_tier'] == tier]

        low = tier_df[tier_df['perp_group'] == 'low (<10)']
        high = tier_df[tier_df['perp_group'] == 'high (>=15)']

        # Get top taxa for this tier
        top_taxa = tier_df['common_taxon'].value_counts().head(10).index.tolist()

        low_counts = low['common_taxon'].value_counts()
        high_counts = high['common_taxon'].value_counts()

        tax_data = []
        for taxon in top_taxa:
            short_name = taxon[:35] + '...' if len(taxon) > 35 else taxon
            tax_data.append({
                'taxon': short_name,
                'low_count': low_counts.get(taxon, 0),
                'high_count': high_counts.get(taxon, 0)
            })

        tax_df = pd.DataFrame(tax_data)

        x = np.arange(len(tax_df))
        width = 0.35

        ax.barh(x - width/2, tax_df['low_count'], width, label='Low (<10)', color='steelblue')
        ax.barh(x + width/2, tax_df['high_count'], width, label='High (>=15)', color='coral')

        ax.set_xlabel('Count')
        ax.set_title(f'Tier {tier} (n={len(tier_df)})', fontsize=12, fontweight='bold')
        ax.set_yticks(x)
        ax.set_yticklabels(tax_df['taxon'], fontsize=9)
        ax.invert_yaxis()

        if idx == 0:
            ax.legend(loc='lower right')

        # Add group sizes
        n_low = len(low)
        n_high = len(high)
        ax.text(0.98, 0.02, f'Low: {n_low}, High: {n_high}',
                transform=ax.transAxes, fontsize=9,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused axes
    for idx in range(n_tiers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Taxonomy Distribution by Perplexity Group (All Tiers, 150M_F)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'taxonomy_all_tiers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved taxonomy_all_tiers.png")


def plot_perplexity_vs_length_all_tiers(df, output_dir):
    """Scatter plot of perplexity vs length for all tiers."""
    tiers = sorted(df['eval_tier'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        tier_df = df[df['eval_tier'] == tier]

        scatter = ax.scatter(
            tier_df['seq_length'],
            tier_df['perplexity'],
            c=tier_df['perplexity'],
            cmap='viridis',
            alpha=0.5,
            s=20
        )

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Perplexity')
        ax.set_title(f'Tier {tier}', fontsize=12, fontweight='bold')

        # Correlation
        valid = tier_df[['seq_length', 'perplexity']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid['seq_length'], valid['perplexity'])
            ax.text(0.05, 0.95, f'r={r:.3f}',
                    transform=ax.transAxes, fontsize=10,
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for idx in range(len(tiers), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Perplexity vs Sequence Length (All Tiers, 150M_F)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity_vs_length_all_tiers.png', dpi=150)
    plt.close()
    print("  Saved perplexity_vs_length_all_tiers.png")


def plot_perplexity_distributions_all_tiers(df, output_dir):
    """Histogram of perplexity distributions for all tiers."""
    tiers = sorted(df['eval_tier'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        tier_df = df[df['eval_tier'] == tier]

        ax.hist(tier_df['perplexity'], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Threshold (10)')
        ax.axvline(x=15, color='orange', linestyle='--', linewidth=2, label='Threshold (15)')

        ax.set_xlabel('Perplexity')
        ax.set_ylabel('Count')
        ax.set_title(f'Tier {tier} (n={len(tier_df)})', fontsize=12, fontweight='bold')

        # Stats
        n_low = len(tier_df[tier_df['perplexity'] < 10])
        n_high = len(tier_df[tier_df['perplexity'] >= 15])
        ax.text(0.98, 0.95, f'<10: {n_low} ({100*n_low/len(tier_df):.1f}%)\n>=15: {n_high} ({100*n_high/len(tier_df):.1f}%)',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    for idx in range(len(tiers), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Perplexity Distributions (All Tiers, 150M_F)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity_distributions_all_tiers.png', dpi=150)
    plt.close()
    print("  Saved perplexity_distributions_all_tiers.png")


def plot_length_by_group_all_tiers(df, output_dir):
    """Box plots of sequence length by perplexity group for all tiers."""
    tiers = sorted(df['eval_tier'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        tier_df = df[df['eval_tier'] == tier]
        tier_filtered = tier_df[tier_df['perp_group'].isin(['low (<10)', 'high (>=15)'])]

        sns.boxplot(data=tier_filtered, x='perp_group', y='seq_length', ax=ax,
                    hue='perp_group', palette='Set2', legend=False)

        ax.set_xlabel('')
        ax.set_ylabel('Sequence Length')
        ax.set_title(f'Tier {tier}', fontsize=12, fontweight='bold')

        # Add medians
        low_med = tier_filtered[tier_filtered['perp_group'] == 'low (<10)']['seq_length'].median()
        high_med = tier_filtered[tier_filtered['perp_group'] == 'high (>=15)']['seq_length'].median()
        ax.text(0.5, 0.95, f'Medians: {low_med:.0f} vs {high_med:.0f}',
                transform=ax.transAxes, fontsize=9, ha='center', va='top')

    for idx in range(len(tiers), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Sequence Length by Perplexity Group (All Tiers, 150M_F)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'length_by_group_all_tiers.png', dpi=150)
    plt.close()
    print("  Saved length_by_group_all_tiers.png")


def analyze_single_tier(df, tier, output_dir, feature_cols):
    """Generate analysis for a single tier."""
    tier_df = df[df['eval_tier'] == tier].copy()
    tier_dir = output_dir / f'tier_{tier}'
    tier_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TIER {tier} ANALYSIS")
    print(f"{'='*60}")

    # Group counts
    print(f"\nPerplexity groups:")
    print(tier_df['perp_group'].value_counts().sort_index())

    # Statistical comparison
    stat_results = statistical_comparison(tier_df, feature_cols)
    stat_results.to_csv(tier_dir / 'statistical_comparison.csv', index=False)

    print("\nTop differentiating features:")
    print(stat_results.head(5).to_string(index=False))

    # Taxonomy summary
    low = tier_df[tier_df['perp_group'] == 'low (<10)']
    high = tier_df[tier_df['perp_group'] == 'high (>=15)']

    print("\nTaxonomy - Low perplexity (<10):")
    for taxon, count in low['common_taxon'].value_counts().head(3).items():
        print(f"  {taxon}: {count} ({100*count/len(low):.1f}%)")

    print("\nTaxonomy - High perplexity (>=15):")
    for taxon, count in high['common_taxon'].value_counts().head(3).items():
        print(f"  {taxon}: {count} ({100*count/len(high):.1f}%)")

    return stat_results


def create_summary_table(df, output_dir):
    """Create a summary table across all tiers."""
    tiers = sorted(df['eval_tier'].unique())

    summary_data = []
    for tier in tiers:
        tier_df = df[df['eval_tier'] == tier]
        low = tier_df[tier_df['perp_group'] == 'low (<10)']
        high = tier_df[tier_df['perp_group'] == 'high (>=15)']

        # Top taxon for each group
        low_top_taxon = low['common_taxon'].value_counts().index[0] if len(low) > 0 else 'N/A'
        low_top_pct = 100 * low['common_taxon'].value_counts().iloc[0] / len(low) if len(low) > 0 else 0
        high_top_taxon = high['common_taxon'].value_counts().index[0] if len(high) > 0 else 'N/A'
        high_top_pct = 100 * high['common_taxon'].value_counts().iloc[0] / len(high) if len(high) > 0 else 0

        summary_data.append({
            'tier': tier,
            'n_total': len(tier_df),
            'n_low': len(low),
            'pct_low': 100 * len(low) / len(tier_df),
            'n_high': len(high),
            'pct_high': 100 * len(high) / len(tier_df),
            'low_median_length': low['seq_length'].median() if len(low) > 0 else 0,
            'high_median_length': high['seq_length'].median() if len(high) > 0 else 0,
            'low_top_taxon': low_top_taxon[:40],
            'low_top_taxon_pct': low_top_pct,
            'high_top_taxon': high_top_taxon[:40],
            'high_top_taxon_pct': high_top_pct,
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'tier_summary.csv', index=False)

    print("\n" + "="*80)
    print("CROSS-TIER SUMMARY")
    print("="*80)
    print(summary_df[['tier', 'n_total', 'n_low', 'pct_low', 'n_high', 'pct_high']].to_string(index=False))

    return summary_df


def main():
    output_dir = Path('plots/all_tiers_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()

    feature_cols = [
        'seq_length', 'shannon_entropy', 'local_complexity',
        'max_homopolymer', 'repeat_fraction', 'hydrophobicity_mean',
        'net_charge', 'fraction_charged', 'member_count',
    ]

    # Generate combined plots
    print("\n" + "="*60)
    print("GENERATING COMBINED VISUALIZATIONS")
    print("="*60)

    plot_perplexity_distributions_all_tiers(df, output_dir)
    plot_combined_taxonomy(df, output_dir)
    plot_perplexity_vs_length_all_tiers(df, output_dir)
    plot_length_by_group_all_tiers(df, output_dir)

    # Analyze each tier
    for tier in sorted(df['eval_tier'].unique()):
        analyze_single_tier(df, tier, output_dir, feature_cols)

    # Create summary table
    summary_df = create_summary_table(df, output_dir)

    print(f"\n\nOutput saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
