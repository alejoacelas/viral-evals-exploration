#!/usr/bin/env python3
"""Process perplexity evaluation files into a single DataFrame."""

import re
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
FILENAME_PATTERN = re.compile(r"given131_(\d+M)_(\w+)_eval_on_(\d+)\.txt")


def parse_header(lines: list[str]) -> dict:
    """Extract metadata from header comment lines."""
    meta = {}
    for line in lines:
        if not line.startswith("#"):
            break
        if ":" in line:
            key, val = line[1:].split(":", 1)
            meta[key.strip()] = val.strip()
    return meta


def parse_filename(fname: str) -> dict | None:
    """Extract model info from filename."""
    match = FILENAME_PATTERN.match(fname)
    if not match:
        return None
    return {
        "param_size": match.group(1),
        "training_tier": match.group(2),
        "eval_tier": int(match.group(3)),
    }


def load_single_file(path: Path) -> pd.DataFrame | None:
    """Load a single evaluation file."""
    file_info = parse_filename(path.name)
    if file_info is None:
        return None

    with open(path) as f:
        lines = f.readlines()

    header_meta = parse_header(lines)

    # Find data start (after header)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("sequence_id"):
            data_start = i
            break

    df = pd.read_csv(path, sep="\t", skiprows=data_start)
    # Add metadata columns
    df["param_size"] = file_info["param_size"]
    df["training_tier"] = file_info["training_tier"]
    df["eval_tier"] = file_info["eval_tier"]
    df["model"] = f"{file_info['param_size']}_{file_info['training_tier']}"
    df["source_file"] = path.name
    return df


def build_combined_dataframe() -> pd.DataFrame:
    """Load all files and combine into single DataFrame."""
    dfs = []
    for path in sorted(OUTPUT_DIR.glob("*.txt")):
        df = load_single_file(path)
        if df is not None:
            dfs.append(df)
            print(f"Loaded {path.name}: {len(df)} sequences")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")
    print(f"Unique sequences: {combined['sequence_id'].nunique()}")
    print(f"Models: {combined['model'].nunique()}")
    return combined


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived columns for analysis."""
    # Numeric param size for sorting
    df["param_millions"] = df["param_size"].str.replace("M", "").astype(int)

    # Whether model was trained on data it's being evaluated on
    # T{n} means tier n was excluded from training
    # So T5 model evaluated on tier 5 = unseen data
    df["is_held_out"] = df.apply(
        lambda r: r["training_tier"] == f"T{r['eval_tier']}", axis=1
    )

    # F = full (trained on all), H = half (trained on random half)
    df["training_type"] = df["training_tier"].apply(
        lambda x: "full" if x == "F" else ("half" if x == "H" else "tier_excluded")
    )

    # Log perplexity for visualization
    df["log_perplexity"] = np.log10(df["perplexity"])
    return df


def main():
    print("Building combined dataset...")
    df = build_combined_dataframe()
    df = add_derived_columns(df)

    # Summary stats
    print("\n--- Summary ---")
    print(f"Shape: {df.shape}")
    print(f"\nParam sizes: {sorted(df['param_size'].unique())}")
    print(f"Training tiers: {sorted(df['training_tier'].unique())}")
    print(f"Eval tiers: {sorted(df['eval_tier'].unique())}")
    print(f"\nPerplexity range: {df['perplexity'].min():.2f} - {df['perplexity'].max():.2f}")
    print(f"Mean perplexity: {df['perplexity'].mean():.2f}")

    # Save
    out_path = Path(__file__).parent / "combined_perplexity.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Also save CSV for inspection
    csv_path = Path(__file__).parent / "combined_perplexity.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    return df


if __name__ == "__main__":
    main()
