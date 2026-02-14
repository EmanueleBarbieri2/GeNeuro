#!/usr/bin/env python3
import subprocess
import sys
import os
import wandb
import re
import argparse
import tempfile

def extract_val(output, pattern):
    matches = re.findall(pattern, output, re.DOTALL | re.MULTILINE)
    if matches:
        val = matches[-1][0] if isinstance(matches[-1], tuple) else matches[-1]
        try: return float(val)
        except: return None
    return None

def main():
    # 1. Initialize WandB (Agent provides the config here)
    run = wandb.init()
    config = wandb.config
    
    # 2. Build the command
    # Using '-u' for unbuffered output to see logs in real-time
    cmd = [sys.executable, '-u', 'model/run_full_pipeline.py']
    
    # Iterate through the config provided by the W&B Sweep
    for key, value in config.items():
        # Handle Boolean Flags (action='store_true')
        if isinstance(value, bool):
            if value is True:
                cmd.append(f'--{key}')
            # If False, we do nothing (the flag remains unset/default)
        else:
            cmd.extend([f'--{key}', str(value)])

    print(f"🚀 Launching ProM3E Pipeline...")
    print(f"📂 Command: {' '.join(cmd)}\n")

    # 3. Transparent Execution
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_log:
        tmp_log_path = tmp_log.name
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tmp_log.write(line)
        
        process.wait()
        tmp_log.seek(0)
        stdout = tmp_log.read()

    if os.path.exists(tmp_log_path):
        os.remove(tmp_log_path)

    # 4. Extract Metrics
    s1_avg = extract_val(stdout, r"Avg: ([\d\.]+)")
    s1_mri = extract_val(stdout, r"MRI➔f: ([\d\.]+)")
    s1_dti = extract_val(stdout, r"DTI➔f: ([\d\.]+)")
    s1_spect = extract_val(stdout, r"SPECT➔f: ([\d\.]+)")
    s2_val = extract_val(stdout, r"Val Loss: ([\d\.]+)")
    s2_mmd = extract_val(stdout, r"MMD Val: ([\d\.]+)")
    s3_acc = extract_val(stdout, r"Val Bal\. Acc: ([\d\.]+)")
    s3_r2_adl = extract_val(stdout, r"prog_U2_ADL\.pt.*?Val R2: ([\d\.\-]+)")
    s3_r2_motor = extract_val(stdout, r"prog_U3_Motor\.pt.*?Val R2: ([\d\.\-]+)")
    s3_stat_adl = extract_val(stdout, r"static_U2_ADL\.pt.*?Val R2: ([\d\.\-]+)")
    s3_stat_motor = extract_val(stdout, r"static_U3_Motor\.pt.*?Val R2: ([\d\.\-]+)")

    # 5. Calculate Final Score
    if s1_avg is not None and s2_val is not None:
        total_score = (0.4 * s1_avg) + (0.6 * s2_val)
    else:
        total_score = 999.0

    # 6. Log everything to the current W&B run
    wandb.log({
        "total_pipeline_score": total_score,
        "stage1/avg_loss": s1_avg,
        "stage1/mri_hub": s1_mri,
        "stage1/dti_hub": s1_dti,
        "stage1/spect_hub": s1_spect,
        "stage2/val_loss": s2_val,
        "stage2/mmd_val": s2_mmd,
        "stage3/classifier_acc": s3_acc,
        "stage3/prog_r2_adl": s3_r2_adl,
        "stage3/prog_r2_motor": s3_r2_motor,
        "stage3/static_r2_adl": s3_stat_adl,
        "stage3/static_r2_motor": s3_stat_motor
    })
    
    print(f"\n✅ Trial Complete. Score: {total_score:.4f}")

if __name__ == "__main__":
    main()