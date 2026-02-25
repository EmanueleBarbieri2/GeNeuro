#!/usr/bin/env python3
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to global CSV")
    parser.add_argument("--topk", type=int, default=15)
    # --> ADDED: New argument to toggle sorting method <--
    parser.add_argument("--sortby", type=str, choices=['none', 'sal', 'z'], default='none',
                        help="Sort results by 'sal' (Saliency), 'z' (Absolute Z-Score), or 'none' (default CSV order)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ File not found: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    
    # --> ADDED: Sorting Logic <--
    if args.sortby == 'sal':
        if 'Global_Saliency_Mean' in df.columns:
            df = df.sort_values(by='Global_Saliency_Mean', ascending=False)
    elif args.sortby == 'z':
        if 'Robustness_Z_Score' in df.columns:
            # Create a temporary column for absolute Z-score to sort by magnitude
            df['abs_z'] = df['Robustness_Z_Score'].abs()
            df = df.sort_values(by='abs_z', ascending=False)
            df = df.drop(columns=['abs_z'])

    # Auto-detect Classification vs Regression
    is_classification = 'PD_Contrib_Mean' in df.columns
    
    # Auto-detect Nodes vs Edges
    if 'Edge_Name' in df.columns:
        name_col = 'Edge_Name'
        feature_type = 'connection'
        name_width = 45 # Edges are long, give them more space
    else:
        name_col = 'ROI_Name'
        feature_type = 'biomarker'
        name_width = 30 # Single ROIs need less space

    # Adjust header width based on content
    total_width = 115 + (name_width - 30)

    print(f"\n{'='*total_width}")
    print(f"🌍 PRO-M3E GLOBAL {feature_type.upper()} RANKING")
    print(f"Target: {os.path.basename(args.csv)}")
    if args.sortby != 'none':
        print(f"Sorted by: {'Absolute Z-Score' if args.sortby == 'z' else 'Saliency'}")
    print(f"{'='*total_width}")
    
    # Reset index so the printed numbering (i+1) is correct for the new sort
    df = df.reset_index(drop=True)
    
    for i, row in df.head(args.topk).iterrows():
        name = str(row.get(name_col, 'Unknown'))[:name_width] # Truncate if insanely long
        mag = row.get('Global_Saliency_Mean', 0.0)
        z_score = row.get('Robustness_Z_Score', 0.0)
        
        if is_classification:
            pd_cont = row.get('PD_Contrib_Mean', 0.0)
            hc_cont = row.get('HC_Contrib_Mean', 0.0)
            prod_cont = row.get('Prodromal_Contrib_Mean', 0.0)
            print(f"{i+1:2}. {name:<{name_width}} | Sal: {mag:.2e} | Z: {z_score:>6.2f} || PD: {pd_cont:>9.2e} | HC: {hc_cont:>9.2e} | Prod: {prod_cont:>9.2e}")
        else:
            global_cont = row.get('Global_Contrib_Mean', 0.0)
            direction = "INCREASES score (Worse)" if global_cont > 0 else "DECREASES score (Better)"
            print(f"{i+1:2}. {name:<{name_width}} | Sal: {mag:.2e} | Z: {z_score:>6.2f} || Impact: {global_cont:>9.2e} -> {direction}")

    print(f"{'='*total_width}")
    
    if not df.empty:
        top_feature = df.iloc[0].get(name_col, 'Top feature')
        print(f"💡 CLINICAL SUMMARY: '{top_feature}' is your most critical {feature_type}.")
        if is_classification:
             print(f"   Look at its PD, HC, and Prod scores to see which diagnosis it strongly points toward.")
        else:
             print(f"   Look at the Impact score to see if it drives the severity/progression up or down.")

if __name__ == "__main__":
    main()