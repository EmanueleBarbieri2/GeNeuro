#!/usr/bin/env python3

import subprocess
import sys
import os
import re
import numpy as np

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, 'run_full_pipeline.py') 
NUM_FOLDS = 5

def extract_metric(pattern, text):
    """Helper to extract float values using robust regex that handles newlines."""
    # re.DOTALL is the key here: it lets '.' match newlines between the header and the value
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return float(match.group(1)) if match else None

def run_fold(fold_idx):
    fold_split_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', f'unified_split_fold{fold_idx}.txt'))
    fold_ckpt_dir = os.path.abspath(os.path.join(BASE_DIR, f'checkpoints_fold{fold_idx}'))
    
    if not os.path.exists(fold_split_path):
         print(f"❌ Error: Missing split file: {fold_split_path}")
         sys.exit(1)

    print(f"\n{'='*50}\n🚀 STARTING FOLD {fold_idx + 1}/{NUM_FOLDS}\n{'='*50}")

    cmd = [sys.executable, PIPELINE_SCRIPT, '--split_path', fold_split_path, '--checkpoints_dir', fold_ckpt_dir]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    full_output = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="") 
        full_output.append(line)
    
    process.stdout.close()
    process.wait()
    stdout = "".join(full_output)

    # --- UPDATED PARSING LOGIC ---
    metrics = {}
    
    # Classification - Simple extraction
    metrics['bal_acc'] = extract_metric(r"Best Balanced Accuracy:\s*([\d.]+)", stdout)
    metrics['f1'] = extract_metric(r"Macro F1-Score:\s*([\d.]+)", stdout)
    metrics['auc'] = extract_metric(r"Macro AUC \(OvR\):\s*([\d.]+)", stdout)

    # Static Regression - Fixed to look past the bracketed target and newlines
    metrics['static_u2_r2'] = extract_metric(r"Static Regression Complete \[UPDRS2_SCORE\].*?Best R2:\s*([-\d.]+)", stdout)
    metrics['static_u3_r2'] = extract_metric(r"Static Regression Complete \[UPDRS3_SCORE\].*?Best R2:\s*([-\d.]+)", stdout)

    # Progression - Fixed to look past the bracketed target and newlines
    metrics['prog_u2_r2'] = extract_metric(r"Progression Training Complete \[UPDRS2_SCORE\].*?Best R2:\s*([-\d.]+)", stdout)
    metrics['prog_u3_r2'] = extract_metric(r"Progression Training Complete \[UPDRS3_SCORE\].*?Best R2:\s*([-\d.]+)", stdout)

    return metrics

if __name__ == '__main__':
    store = {k: [] for k in ['bal_acc', 'f1', 'auc', 'static_u2_r2', 'static_u3_r2', 'prog_u2_r2', 'prog_u3_r2']}
    
    for i in range(NUM_FOLDS):
        m = run_fold(i)
        
        # Check if we at least got the main accuracy or one regression value
        if m['bal_acc'] is not None or m['static_u3_r2'] is not None:
            for key in store:
                if m[key] is not None: 
                    store[key].append(m[key])
        else:
            print(f"⚠️ Warning: Fold {i} failed to parse key metrics.")

    # --- FINAL STATISTICAL TABLE ---
    print(f"\n{'='*60}\n🏆 5-FOLD CROSS-VALIDATION FINAL RESULTS 🏆\n{'='*60}")
    
    def print_stat(label, data_list):
        if data_list:
            print(f"{label:<25}: {np.mean(data_list):.4f} ± {np.std(data_list):.4f} (n={len(data_list)})")

    print("🧠 CLASSIFICATION:")
    print_stat("  Balanced Accuracy", store['bal_acc'])
    print_stat("  Macro F1-Score", store['f1'])
    print_stat("  Macro AUC", store['auc'])

    print("\n📈 STATIC UPDRS (R2):")
    print_stat("  UPDRS2 (ADL)", store['static_u2_r2'])
    print_stat("  UPDRS3 (Motor)", store['static_u3_r2'])

    print("\n⏳ PROGRESSION (R2):")
    print_stat("  UPDRS2 (ADL)", store['prog_u2_r2'])
    print_stat("  UPDRS3 (Motor)", store['prog_u3_r2'])
    print(f"{'='*60}\n")