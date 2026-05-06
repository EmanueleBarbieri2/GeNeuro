#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to global CSV")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--sortby", type=str, choices=['none', 'sal', 'z', 'rsal', 'mult'], default='none',
                        help="Sort by: 'sal' (Mean), 'z' (Z-Score), 'rsal' (Relative %), 'mult' (Multiplier), or 'none'")
    parser.add_argument("--include_self_loops", action="store_true", help="Include nodal self-connections (e.g., ROI <-> ROI)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ File not found: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    
    # --- 1. FILTER SELF-LOOPS (NODAL IMPORTANCE) ---
    # We do this first so the median/sum stats are based ONLY on inter-regional connections
    is_edge_file = 'Edge_Name' in df.columns
    
    if is_edge_file and not args.include_self_loops:
        initial_count = len(df)
        # Assumes format "Region A <-> Region B"
        # We split the name and check if both sides are identical
        df = df[df['Edge_Name'].apply(lambda x: len(set(str(x).split(' <-> '))) > 1)].copy()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"✂️ Filtered out {removed} self-loops (Nodal Importance) to focus on inter-regional tracts.")

    # --- 2. CALCULATE MEDIAN & MULTIPLIERS ---
    if 'Global_Saliency_Mean' in df.columns:
        # Calculate global median of the remaining (inter-regional) features
        median_sal = df['Global_Saliency_Mean'].median()
        total_sal = df['Global_Saliency_Mean'].sum()
        
        safe_median = median_sal if median_sal > 0 else 1e-12
        
        df['Rel_Sal_Pct'] = (df['Global_Saliency_Mean'] / total_sal) * 100
        df['Saliency_Mult'] = df['Global_Saliency_Mean'] / safe_median
    else:
        df['Rel_Sal_Pct'] = 0.0
        df['Saliency_Mult'] = 0.0
        median_sal = 0.0

    # --- 3. SORTING LOGIC ---
    if args.sortby in ['sal', 'mult']:
        df = df.sort_values(by='Global_Saliency_Mean', ascending=False)
    elif args.sortby == 'rsal':
        df = df.sort_values(by='Rel_Sal_Pct', ascending=False)
    elif args.sortby == 'z':
        if 'Robustness_Z_Score' in df.columns:
            df['abs_z'] = df['Robustness_Z_Score'].abs()
            df = df.sort_values(by='abs_z', ascending=False)
            df = df.drop(columns=['abs_z'])

    # Auto-detect formatting
    is_classification = 'PD_Contrib_Mean' in df.columns
    name_col = 'Edge_Name' if is_edge_file else 'ROI_Name'
    feature_type = 'connection' if is_edge_file else 'biomarker'
    name_width = 50 if is_edge_file else 30
    total_width = 135 + (name_width - 30)

    print(f"\n{'='*total_width}")
    print(f"🌍 PRO-M3E GLOBAL {feature_type.upper()} RANKING (Median Saliency: {median_sal:.2e})")
    print(f"Target: {os.path.basename(args.csv)}")
    if is_edge_file and not args.include_self_loops:
        print(f"Mode: Inter-Regional Connections Only")
    print(f"{'='*total_width}")
    
    df = df.reset_index(drop=True)
    
    for i, row in df.head(args.topk).iterrows():
        name = str(row.get(name_col, 'Unknown'))[:name_width]
        rel_sal = row.get('Rel_Sal_Pct', 0.0)
        multiplier = row.get('Saliency_Mult', 0.0)
        z_score = row.get('Robustness_Z_Score', 0.0)
        
        line_start = f"{i+1:2}. {name:<{name_width}} | {multiplier:>6.1f}x | Rel: {rel_sal:>5.1f}% | Z: {z_score:>6.2f}"
        
        if is_classification:
            pd_cont = row.get('PD_Contrib_Mean', 0.0)
            hc_cont = row.get('HC_Contrib_Mean', 0.0)
            # Add Prodromal if it exists in the CSV
            if 'Prodromal_Contrib_Mean' in df.columns:
                prod_cont = row.get('Prodromal_Contrib_Mean', 0.0)
                print(f"{line_start} || PD: {pd_cont:>8.1e} | HC: {hc_cont:>8.1e} | PR: {prod_cont:>8.1e}")
            else:
                print(f"{line_start} || PD: {pd_cont:>9.2e} | HC: {hc_cont:>9.2e}")
        else:
            global_cont = row.get('Global_Contrib_Mean', 0.0)
            direction = "+" if global_cont > 0 else "-"
            print(f"{line_start} || Impact: {global_cont:>9.2e} ({direction})")

    print(f"{'='*total_width}")
    
    if not df.empty:
        top_feature = df.iloc[0].get(name_col, 'Top feature')
        top_mult = df.iloc[0].get('Saliency_Mult', 0.0)
        print(f"💡 DOMINANCE SUMMARY: '{top_feature}' is {top_mult:.1f} times more salient than the median.")
        print(f"   This represents the most significant {feature_type} in the validated cohort.")

if __name__ == "__main__":
    main()