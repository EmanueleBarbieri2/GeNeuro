import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold

# --- Configuration ---
CSV_PATH = "/home/ebarbieri/GeNeuro_HPC/data/PPMI_Curated_Data_Cut_Public_20251112.csv"
OUTPUT_DIR = "." # Saves in the same directory
NUM_FOLDS = 5

def create_folds():
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Ensure our target and group columns are clean
    df['PATNO'] = df['PATNO'].astype(str)
    df['COHORT'] = df['COHORT'].astype(int)
    
    # Initialize StratifiedGroupKFold
    # shuffle=True ensures we don't just split chronologically by PATNO
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # We will use COHORT for stratification, and PATNO for grouping
    fold_generator = sgkf.split(X=df, y=df['COHORT'], groups=df['PATNO'])
    
    for fold, (train_idx, val_idx) in enumerate(fold_generator):
        print(f"\n--- Generating Fold {fold} ---")
        
        # Create a new column specifically for this split
        split_df = df[['PATNO', 'EVENT_ID', 'COHORT']].copy()
        split_df['split'] = 'none'
        
        # Assign train and val tags
        split_df.loc[train_idx, 'split'] = 'train'
        split_df.loc[val_idx, 'split'] = 'val'
        
        # --- Sanity Checks to prove it worked ---
        train_patients = set(split_df[split_df['split'] == 'train']['PATNO'])
        val_patients = set(split_df[split_df['split'] == 'val']['PATNO'])
        overlap = train_patients.intersection(val_patients)
        
        assert len(overlap) == 0, f"🚨 LEAKAGE DETECTED in Fold {fold}: {overlap}"
        
        train_dist = split_df[split_df['split'] == 'train']['COHORT'].value_counts(normalize=True).to_dict()
        val_dist = split_df[split_df['split'] == 'val']['COHORT'].value_counts(normalize=True).to_dict()
        
        print(f"Leakage Check: Passed! (0 overlapping patients)")
        print(f"Train Size: {len(train_idx)} rows | Val Size: {len(val_idx)} rows")
        print(f"Train Class Distribution: {train_dist}")
        print(f"Val Class Distribution:   {val_dist}")
        
        # --- Save to TXT ---
        # Assuming your unified_split.txt is formatted as "PATNO_EVENT_ID,split"
        # Update this formatting if your dataloader expects something different!
        # --- Save to TXT (Updated for your train.py parser) ---
        output_path = os.path.join(OUTPUT_DIR, f"unified_split_fold{fold}.txt")
        
        train_subset = split_df[split_df['split'] == 'train']
        val_subset = split_df[split_df['split'] == 'val']
        
        with open(output_path, 'w') as f:
            # 1. Write the Train IDs header and list
            f.write("train_ids:\n")
            for _, row in train_subset.iterrows():
                f.write(f"{row['PATNO']}_{row['EVENT_ID']}\n")
                
            # 2. Write the Val IDs header and list
            f.write("val_ids:\n")
            for _, row in val_subset.iterrows():
                f.write(f"{row['PATNO']}_{row['EVENT_ID']}\n")
                
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    create_folds()