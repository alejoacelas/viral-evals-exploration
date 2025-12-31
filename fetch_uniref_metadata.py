"""
Fetch UniRef50 metadata from UniProt REST API for Tier 4 sequences.

This script fetches sequence metadata and computes sequence-level properties
for analysis of perplexity differences.
"""

import pandas as pd
import requests
import time
import math
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# UniProt REST API base URL
UNIREF_API = "https://rest.uniprot.org/uniref"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'U': 2.5, 'O': -3.9, 'B': -3.5, 'Z': -3.5, 'X': 0.0
}

# Amino acid charges at pH 7
CHARGE = {
    'K': 1, 'R': 1, 'H': 0.1,  # positive
    'D': -1, 'E': -1,          # negative
}


def compute_shannon_entropy(sequence: str) -> float:
    """Compute Shannon entropy of amino acid composition."""
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    length = len(sequence)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    return entropy


def compute_local_complexity(sequence: str, window_size: int = 12) -> float:
    """
    Compute average local sequence complexity using sliding window entropy.
    Low values indicate low-complexity regions.
    """
    if len(sequence) < window_size:
        return compute_shannon_entropy(sequence)

    entropies = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        entropies.append(compute_shannon_entropy(window))

    return sum(entropies) / len(entropies) if entropies else 0.0


def compute_max_homopolymer(sequence: str) -> int:
    """Find the length of the longest homopolymer (consecutive same AA)."""
    if not sequence:
        return 0
    max_len = 1
    current_len = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 1
    return max_len


def compute_repeat_fraction(sequence: str, min_repeat: int = 2) -> float:
    """
    Compute fraction of sequence covered by short tandem repeats.
    Looks for di-, tri-, and tetrapeptide repeats.
    """
    if len(sequence) < 4:
        return 0.0

    repeat_positions = set()

    # Check for di, tri, and tetrapeptide repeats
    for unit_len in [2, 3, 4]:
        for i in range(len(sequence) - unit_len):
            unit = sequence[i:i + unit_len]
            repeat_count = 1
            j = i + unit_len
            while j + unit_len <= len(sequence) and sequence[j:j + unit_len] == unit:
                repeat_count += 1
                j += unit_len

            if repeat_count >= min_repeat:
                for pos in range(i, j):
                    repeat_positions.add(pos)

    return len(repeat_positions) / len(sequence) if sequence else 0.0


def compute_sequence_properties(sequence: str) -> dict:
    """Compute various sequence-level properties."""
    if not sequence:
        return {
            'seq_length': 0,
            'shannon_entropy': 0.0,
            'local_complexity': 0.0,
            'max_homopolymer': 0,
            'repeat_fraction': 0.0,
            'hydrophobicity_mean': 0.0,
            'net_charge': 0.0,
            'fraction_charged': 0.0,
        }

    # Basic properties
    length = len(sequence)

    # Complexity measures
    entropy = compute_shannon_entropy(sequence)
    local_complexity = compute_local_complexity(sequence)
    max_homopolymer = compute_max_homopolymer(sequence)
    repeat_fraction = compute_repeat_fraction(sequence)

    # Hydrophobicity
    hydro_values = [HYDROPHOBICITY.get(aa, 0.0) for aa in sequence]
    hydrophobicity_mean = sum(hydro_values) / length if length > 0 else 0.0

    # Charge
    net_charge = sum(CHARGE.get(aa, 0) for aa in sequence)
    charged_count = sum(1 for aa in sequence if aa in CHARGE)
    fraction_charged = charged_count / length if length > 0 else 0.0

    # Amino acid composition (fraction of each standard AA)
    aa_counts = Counter(sequence)
    aa_fractions = {f'frac_{aa}': aa_counts.get(aa, 0) / length
                    for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    return {
        'seq_length': length,
        'shannon_entropy': entropy,
        'local_complexity': local_complexity,
        'max_homopolymer': max_homopolymer,
        'repeat_fraction': repeat_fraction,
        'hydrophobicity_mean': hydrophobicity_mean,
        'net_charge': net_charge,
        'fraction_charged': fraction_charged,
        **aa_fractions
    }


def fetch_uniref_entry(cluster_id: str, max_retries: int = 3) -> dict:
    """
    Fetch a UniRef50 cluster entry from UniProt REST API.
    Returns cluster metadata including representative sequence.
    """
    url = f"{UNIREF_API}/{cluster_id}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()

                # Extract relevant fields
                rep_member = data.get('representativeMember', {})
                sequence_info = rep_member.get('sequence', {})

                return {
                    'cluster_id': cluster_id,
                    'name': data.get('name', ''),
                    'member_count': data.get('memberCount', 0),
                    'common_taxon': data.get('commonTaxon', {}).get('scientificName', ''),
                    'common_taxon_id': data.get('commonTaxon', {}).get('taxonId', ''),
                    'sequence': sequence_info.get('value', ''),
                    'seq_length_api': sequence_info.get('length', 0),
                    'rep_accession': rep_member.get('memberId', ''),
                    'rep_organism': rep_member.get('organismName', ''),
                    'seed_id': data.get('seedId', ''),
                }
            elif response.status_code == 404:
                print(f"  Not found: {cluster_id}")
                return {'cluster_id': cluster_id, 'error': 'not_found'}
            else:
                print(f"  HTTP {response.status_code} for {cluster_id}, retrying...")
                time.sleep(1)
        except requests.RequestException as e:
            print(f"  Request error for {cluster_id}: {e}, retrying...")
            time.sleep(1)

    return {'cluster_id': cluster_id, 'error': 'max_retries'}


def fetch_all_metadata(sequence_ids: list, max_workers: int = 10) -> list:
    """
    Fetch metadata for all sequences using parallel requests.
    """
    results = []
    total = len(sequence_ids)

    print(f"Fetching metadata for {total} sequences...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(fetch_uniref_entry, seq_id): seq_id
            for seq_id in sequence_ids
        }

        completed = 0
        for future in as_completed(future_to_id):
            result = future.result()
            results.append(result)
            completed += 1

            if completed % 100 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    return results


def main():
    # Load perplexity data
    print("Loading perplexity data...")
    df = pd.read_csv('combined_perplexity.csv')

    # Get all unique sequences across all tiers
    all_sequences = df['sequence_id'].unique().tolist()
    print(f"Found {len(all_sequences)} unique sequences across all tiers")

    # Check if we already have cached metadata
    cache_path = Path('uniref_metadata.csv')
    if cache_path.exists():
        print(f"Loading cached metadata from {cache_path}...")
        cached = pd.read_csv(cache_path)
        cached_ids = set(cached['sequence_id'].tolist())
        missing_ids = [sid for sid in all_sequences if sid not in cached_ids]
        print(f"  Cached: {len(cached_ids)}, Missing: {len(missing_ids)}")

        if not missing_ids:
            print("All metadata already cached!")
            return cached
    else:
        missing_ids = all_sequences
        cached = None

    # Fetch missing metadata
    if missing_ids:
        raw_results = fetch_all_metadata(missing_ids)

        # Process results and compute sequence properties
        print("\nComputing sequence properties...")
        processed_results = []
        for raw in raw_results:
            if 'error' in raw:
                processed_results.append({
                    'sequence_id': raw['cluster_id'],
                    'error': raw['error']
                })
                continue

            sequence = raw.get('sequence', '')
            seq_props = compute_sequence_properties(sequence)

            processed_results.append({
                'sequence_id': raw['cluster_id'],
                'name': raw.get('name', ''),
                'member_count': raw.get('member_count', 0),
                'common_taxon': raw.get('common_taxon', ''),
                'common_taxon_id': raw.get('common_taxon_id', ''),
                'rep_accession': raw.get('rep_accession', ''),
                'rep_organism': raw.get('rep_organism', ''),
                'sequence': sequence,
                **seq_props
            })

        new_df = pd.DataFrame(processed_results)

        # Merge with cached if exists
        if cached is not None:
            metadata_df = pd.concat([cached, new_df], ignore_index=True)
        else:
            metadata_df = new_df

        # Save to cache
        print(f"\nSaving metadata to {cache_path}...")
        metadata_df.to_csv(cache_path, index=False)
        print(f"Saved {len(metadata_df)} entries")
    else:
        metadata_df = cached

    # Print summary stats
    print("\n" + "="*50)
    print("METADATA SUMMARY")
    print("="*50)

    valid_df = metadata_df[metadata_df['error'].isna()] if 'error' in metadata_df.columns else metadata_df

    print(f"Total sequences: {len(metadata_df)}")
    print(f"Successfully fetched: {len(valid_df)}")

    if len(valid_df) > 0:
        print(f"\nSequence length stats:")
        print(f"  Min: {valid_df['seq_length'].min()}")
        print(f"  Max: {valid_df['seq_length'].max()}")
        print(f"  Mean: {valid_df['seq_length'].mean():.1f}")
        print(f"  Median: {valid_df['seq_length'].median():.1f}")

        print(f"\nTop 10 common taxa:")
        tax_counts = valid_df['common_taxon'].value_counts().head(10)
        for taxon, count in tax_counts.items():
            print(f"  {taxon}: {count}")

    return metadata_df


if __name__ == '__main__':
    main()
