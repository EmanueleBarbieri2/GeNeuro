#!/usr/bin/env python3
"""
Master ProM3E Pipeline:
1. Stage 1: Contrastive Alignment (Encoder Training)
2. Stage 2: Generative Hallucination (ProM3E Generator)
3. Stage 3: Specialist Downstream Tasks (Decoupled Targets)
"""

import subprocess
import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Full ProM3E pipeline with Specialist Tuning.")
    
    # --- Paths & General ---
    parser.add_argument('--data_csv', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'PPMI_Curated_Data_Cut_Public_20251112.csv')))
    parser.add_argument('--split_path', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_split.txt')))
    parser.add_argument('--checkpoints_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints')))
    parser.add_argument('--device', default='cuda' if os.path.exists('/dev/nvidia0') else 'cpu')
    
    # --- Stage 1: Contrastive Pre-training ---
    parser.add_argument('--contrastive_epochs', type=int, default=100)
    parser.add_argument('--contrastive_lr', type=float, default=1e-4)
    parser.add_argument('--contrastive_alpha', type=float, default=-5.0)
    parser.add_argument('--contrastive_beta', type=float, default=5.0)
    parser.add_argument('--no_contrastive_aug', action='store_true')
    
    # --- Stage 2: ProM3E Generator ---
    parser.add_argument('--generator_epochs', type=int, default=100)
    parser.add_argument('--generator_lr', type=float, default=1e-4)
    parser.add_argument('--generator_alpha', type=float, default=-5.0)
    parser.add_argument('--generator_beta', type=float, default=5.0)
    parser.add_argument('--generator_lambda', type=float, default=0.001)
    parser.add_argument('--no_aug', action='store_true')
    
    # --- Stage 3: Downstream Tasks ---
    parser.add_argument('--cls_epochs', type=int, default=100)
    parser.add_argument('--prog_epochs', type=int, default=100)
    parser.add_argument('--updrs_epochs', type=int, default=100)
    
    return parser.parse_args()

args = parse_args()

# --- Path Management ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = args.checkpoints_dir
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DATA_CSV = args.data_csv
SPLIT_PATH = args.split_path
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

# Shared Resources
EMBEDDINGS_PATH = os.path.join(CHECKPOINTS_DIR, 'embeddings.pt')
ENCODER_CKPT = os.path.join(CHECKPOINTS_DIR, 'encoders.pt')
GENERATOR_CKPT = os.path.join(CHECKPOINTS_DIR, 'prom3e_generator.pt')

# Shared Environment
ENV = os.environ.copy()
ENV['PYTHONPATH'] = '.'

def run_contrastive():
    print(f"\n🚀 STAGE 1: Contrastive Alignment (Hub: fMRI)")
    cmd = [
        sys.executable, 'model/contrastive/train.py',
        '--epochs', str(args.contrastive_epochs),
        '--lr', str(args.contrastive_lr),
        '--alpha', str(args.contrastive_alpha),
        '--beta', str(args.contrastive_beta),
        '--embeddings_path', EMBEDDINGS_PATH,
        '--encoders_path', ENCODER_CKPT,
        '--split_path', SPLIT_PATH,
        '--data_root', DATA_ROOT,
        '--device', args.device
    ]
    if args.no_contrastive_aug: cmd.append('--no_aug')
    subprocess.run(cmd, check=True, env=ENV)

def train_generator():
    print(f"\n🚀 STAGE 2: ProM3E Generator (Depth: 3 Layers)")
    cmd = [
        sys.executable, 'model/generator/train_generator.py',
        '--csv_path', DATA_CSV,
        '--out', GENERATOR_CKPT,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--split_path', SPLIT_PATH,
        '--epochs', str(args.generator_epochs),
        '--lr', str(args.generator_lr),
        '--alpha', str(args.generator_alpha),
        '--beta', str(args.generator_beta),
        '--lambd', str(args.generator_lambda),
        '--device', args.device
    ]
    if args.no_aug: cmd.append('--no_aug')
    subprocess.run(cmd, check=True, env=ENV)

def run_downstream():
    print('\n📊 STAGE 3: Granular Downstream Tasks')
    
    # 1. Classification (Healthy vs PD vs Prodromal)
    print('>> Running Global Classification...')
    subprocess.run([
        sys.executable, 'model/downstream/downstream_classification.py',
        '--epochs', str(args.cls_epochs),
        '--csv_path', DATA_CSV,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--device', args.device
    ], check=True, env=ENV)

    # 2. Decoupled Progression Specialists (Temporal)
    # Target Indices: 1=UPDRS2 (ADL), 2=UPDRS3 (Motor)
    for idx, name in [(1, 'U2_ADL'), (2, 'U3_Motor')]:
        print(f'>> Training Temporal Progression Specialist: {name}')
        subprocess.run([
            sys.executable, 'model/downstream/downstream_progression.py',
            '--target_idx', str(idx),
            '--progression_ckpt', os.path.join(CHECKPOINTS_DIR, f'prog_{name}.pt'),
            '--epochs', str(args.prog_epochs),
            '--csv_path', DATA_CSV,
            '--embeddings_path', EMBEDDINGS_PATH,
            '--generator_ckpt', GENERATOR_CKPT,
            '--device', args.device
        ], check=True, env=ENV)

    # 3. Decoupled Static Specialists (Per-Visit)
    for idx, name in [(1, 'U2_ADL'), (2, 'U3_Motor')]:
        print(f'>> Training Static UPDRS Specialist: {name}')
        subprocess.run([
            sys.executable, 'model/downstream/downstream_updrs.py',
            '--target_idx', str(idx),
            '--updrs_ckpt', os.path.join(CHECKPOINTS_DIR, f'static_{name}.pt'),
            '--epochs', str(args.updrs_epochs),
            '--csv_path', DATA_CSV,
            '--embeddings_path', EMBEDDINGS_PATH,
            '--generator_ckpt', GENERATOR_CKPT,
            '--device', args.device
        ], check=True, env=ENV)

if __name__ == '__main__':
    run_contrastive()
    train_generator()
    run_downstream()
    print('\n✅ Full ProM3E Pipeline Complete. All Specialist Models saved.')