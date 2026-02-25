import pandas as pd
import os

CSV_PATH = "/home/ebarbieri/GeNeuro_HPC/data/PPMI_Curated_Data_Cut_Public_20251112.csv"
OUTPUT_PATH = "unified_split_master.txt"

def create_master_split():
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Get all unique Patient-Visit combinations
    all_visits = (df['PATNO'].astype(str) + "_" + df['EVENT_ID'].astype(str)).unique()
    
    with open(OUTPUT_PATH, 'w') as f:
        # Write 100% of data to train
        f.write("train_ids:\n")
        for visit in all_visits:
            f.write(f"{visit}\n")
            
        # Write 100% of data to val (to prevent division by zero and trigger saves)
        f.write("val_ids:\n")
        for visit in all_visits:
            f.write(f"{visit}\n")
            
    print(f"✅ Created Master Split with {len(all_visits)} visits in both Train and Val!")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    create_master_split()