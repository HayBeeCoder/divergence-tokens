#!/usr/bin/env python3
"""
Multi-hop preference transfer analysis.
Computes relative change, normalized change, trend direction, and saturation detection.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path("mean-log-prob")
OUTDIR.mkdir(exist_ok=True)

# Read CSV
df = pd.read_csv("mean-log-prob/mean_by_hop.csv")

# Filter for checkpoints only
df_ckpt = df[df['model_tag'].str.startswith('checkpoint')].copy()
df_ckpt['hop_num'] = df_ckpt['hop'].str.extract('(\d+)').astype(int)
df_ckpt = df_ckpt.sort_values('hop_num')

# Separate by condition (withprompt, noprompt)
conditions = df_ckpt['hop'].str.extract('(withprompt|noprompt)')[0].unique()

for condition in conditions:
    subset = df_ckpt[df_ckpt['hop'].str.contains(condition)].sort_values('hop_num').copy()
    
    if len(subset) < 2:
        print(f"Skipping {condition}: insufficient data")
        continue
    
    print(f"\n{'='*70}")
    print(f"Analysis for: {condition}")
    print(f"{'='*70}")
    
    # Baseline: use first hop as H0
    h0_idx = subset.index[0]
    h0_value = subset.loc[h0_idx, 'mean']
    h0_hop = subset.loc[h0_idx, 'hop']
    
    print(f"Baseline (H₀) = {h0_hop}: {h0_value:.6f}\n")
    
    # Compute metrics
    subset['relative_change'] = subset['mean'] - h0_value
    subset['normalized_change'] = (subset['mean'] - h0_value) / abs(h0_value)
    
    # Identify trend and saturation
    subset['improvement'] = subset['relative_change']  # Higher (less negative) is better
    subset['improvement_pct'] = (subset['improvement'] / abs(h0_value)) * 100
    
    # Compute second derivative to detect saturation
    if len(subset) >= 3:
        subset['delta_improvement'] = subset['improvement'].diff()
        subset['accel'] = subset['delta_improvement'].diff()  # second derivative
    
    # Print table
    print("Metrics table:")
    print("-" * 120)
    print(f"{'Hop':<20} {'Log P':<15} {'Relative':<15} {'Normalized':<15} {'Improve %':<15} {'Δ Improve':<15}")
    print("-" * 120)
    for idx, row in subset.iterrows():
        delta_str = f"{row['delta_improvement']:.6f}" if pd.notna(row['delta_improvement']) else "N/A"
        print(f"{row['hop']:<20} {row['mean']:<15.6f} {row['relative_change']:<15.6f} {row['normalized_change']:<15.6f} {row['improvement_pct']:<15.2f} {delta_str:<15}")
    print("-" * 120)
    
    # Trend direction
    trend_vals = subset['improvement'].values
    if len(trend_vals) >= 2:
        trend = "increasing" if trend_vals[-1] > trend_vals[0] else "decreasing"
        max_improvement = subset.loc[subset['improvement'].idxmax()]
        print(f"\nTrend direction: {trend.upper()}")
        print(f"Best performance: {max_improvement['hop']} with improvement of {max_improvement['improvement_pct']:.2f}%")
    
    # Saturation detection
    if 'accel' in subset.columns and subset['accel'].notna().sum() > 0:
        accel_vals = subset['accel'].dropna().values
        if len(accel_vals) > 0:
            mean_accel = np.mean(accel_vals)
            recent_accel = accel_vals[-1]
            if recent_accel > -1e-6:  # close to zero
                print(f"⚠ Saturation detected: acceleration = {recent_accel:.6f} (near zero)")
                print(f"  Performance is plateauing — diminishing returns observed")
            else:
                print(f"📈 Still improving: acceleration = {recent_accel:.6f}")
    
    # ============ PLOTS ============
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Absolute log P (reference)
    ax = axes[0, 0]
    ax.plot(range(len(subset)), subset['mean'], marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(h0_value, color='red', linestyle='--', label=f'Baseline {h0_hop}', linewidth=2)
    ax.set_xlabel('Hop Index', fontsize=11)
    ax.set_ylabel('Mean log P(target)', fontsize=11)
    ax.set_title(f'Absolute log P — {condition}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels([h.replace('_withprompt','').replace('_noprompt','') for h in subset['hop']], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Relative change from baseline
    ax = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in subset['relative_change']]
    ax.bar(range(len(subset)), subset['relative_change'], color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Relative Change (log P - baseline)', fontsize=11)
    ax.set_title(f'Relative Change from Baseline — {condition}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels([h.replace('_withprompt','').replace('_noprompt','') for h in subset['hop']], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(subset['relative_change']):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    # Plot 3: Normalized change (%)
    ax = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in subset['normalized_change']]
    ax.bar(range(len(subset)), subset['normalized_change']*100, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Normalized Change (%)', fontsize=11)
    ax.set_title(f'Normalized Change: (log P - baseline) / |baseline| — {condition}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels([h.replace('_withprompt','').replace('_noprompt','') for h in subset['hop']], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(subset['normalized_change']*100):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    # Plot 4: Improvement % trend with acceleration overlay
    ax = axes[1, 1]
    ax.plot(range(len(subset)), subset['improvement_pct'], marker='o', linewidth=2.5, markersize=9, 
            color='darkgreen', label='Improvement %')
    if 'delta_improvement' in subset.columns:
        # Plot delta (rate of change) on secondary axis
        ax2 = ax.twinx()
        delta_vals = subset.loc[subset['delta_improvement'].notna(), 'delta_improvement'].values
        delta_indices = subset['delta_improvement'].notna().cumsum() - 1
        delta_indices = delta_indices[delta_indices >= 0].unique()
        if len(delta_vals) > 0:
            ax2.plot(range(1, len(delta_vals)+1), delta_vals, 
                    marker='s', linestyle='--', linewidth=2, markersize=7, color='orange', alpha=0.7, label='Rate of change')
        ax2.axhline(0, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Rate of Change (Δ improvement)', fontsize=11, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    
    ax.set_xlabel('Hop', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11, color='darkgreen')
    ax.set_title(f'Improvement Trend & Saturation Detection — {condition}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels([h.replace('_withprompt','').replace('_noprompt','') for h in subset['hop']], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    if 'delta_improvement' in subset.columns:
        ax2.legend(loc='upper right')
    
    # Add value labels
    for i, v in enumerate(subset['improvement_pct']):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, color='darkgreen', fontweight='bold')
    
    plt.tight_layout()
    fname = OUTDIR / f"multihop_analysis_{condition}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved: {fname}")
    
    # CSV export
    export_df = subset[['hop', 'hop_num', 'mean', 'relative_change', 'normalized_change', 'improvement_pct']].copy()
    csv_fname = OUTDIR / f"multihop_metrics_{condition}.csv"
    export_df.to_csv(csv_fname, index=False)
    print(f"✓ CSV saved: {csv_fname}")

print(f"\n{'='*70}")
print("All analyses complete.")
