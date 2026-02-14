#!/usr/bin/env python3
"""
Master ProM3E Pipeline:
1. Stage 1: Contrastive Alignment (Encoder Training)
2. Stage 2: Generative Hallucination (MMD/KL Toggle & Many-to-Many Masking)
3. INTERMEDIATE: Smart Hybrid Reconstruction (Real + Hallucinated Fusion)
4. Stage 3: Specialist Downstream Tasks (Decoupled Targets)
"""

import subprocess
import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Full ProM3E pipeline with Smart Hybrid Fusion.")
    
    # --- Paths & General ---
    parser.add_argument('--data_csv', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'PPMI_Curated_Data_Cut_Public_20251112.csv')))
    parser.add_argument('--split_path', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_split.txt')))
    parser.add_argument('--checkpoints_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints')))
    parser.add_argument('--device', default='cpu')
    
    # --- Stage 1: Contrastive Pre-training ---
    parser.add_argument('--contrastive_epochs', type=int, default=100)
    parser.add_argument('--contrastive_lr', type=float, default=5e-4)
    parser.add_argument('--contrastive_batch_size', type=int, default=32) # Added
    parser.add_argument('--contrastive_alpha', type=float, default=-5.0)
    parser.add_argument('--contrastive_beta', type=float, default=5.0)
    parser.add_argument('--hub_name', default='fMRI', choices=['fMRI', 'MRI', 'SPECT', 'DTI'])
    parser.add_argument('--aug_mask', type=float, default=0.15)
    parser.add_argument('--aug_jitter', type=float, default=0.01)
    parser.add_argument('--clip_val', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=0.80)
    parser.add_argument('--no_contrastive_aug', action='store_true')
    
    # --- Stage 2: ProM3E Generator ---
    parser.add_argument('--generator_epochs', type=int, default=100)
    parser.add_argument('--generator_lr', type=float, default=1e-4)
    parser.add_argument('--generator_weight_decay', type=float, default=1e-4)
    parser.add_argument('--generator_alpha', type=float, default=-5.0)
    parser.add_argument('--generator_beta', type=float, default=5.0)
    parser.add_argument('--generator_lambda', type=float, default=0.001)
    parser.add_argument('--generator_divergence', choices=['mmd', 'kl', 'none'], default='mmd')
    parser.add_argument('--gen_keep_prob', type=float, default=0.5)
    parser.add_argument('--gen_kl_warmup', type=int, default=10)
    parser.add_argument('--gen_hidden_dim', type=int, default=512)
    parser.add_argument('--gen_num_heads', type=int, default=8)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--gen_num_registers', type=int, default=4)
    parser.add_argument('--gen_mlp_depth', type=int, default=2)
    parser.add_argument('--gen_dropout', type=float, default=0.1)
    
    # --- Stage 3: Downstream Tasks ---
    parser.add_argument('--downstream_lr', type=float, default=1e-3) # Added
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

EMBEDDINGS_PATH = os.path.join(CHECKPOINTS_DIR, 'embeddings.pt')
ENCODER_CKPT = os.path.join(CHECKPOINTS_DIR, 'encoders.pt')
GENERATOR_CKPT = os.path.join(CHECKPOINTS_DIR, 'prom3e_generator.pt')
RECON_DEMO_PATH = os.path.join(CHECKPOINTS_DIR, 'recon_demo.pt')

ENV = os.environ.copy()
ENV['PYTHONPATH'] = '.'

def run_contrastive():
    print(f"\n🚀 STAGE 1: Contrastive Alignment (Hub: {args.hub_name})")
    cmd = [
        sys.executable, 'model/contrastive/train.py',
        '--epochs', str(args.contrastive_epochs),
        '--lr', str(args.contrastive_lr),
        '--batch_size', str(args.contrastive_batch_size),
        '--alpha', str(args.contrastive_alpha),
        '--beta', str(args.contrastive_beta),
        '--hub_name', args.hub_name,
        '--aug_mask', str(args.aug_mask),
        '--aug_jitter', str(args.aug_jitter),
        '--clip_val', str(args.clip_val),
        '--hidden_dim', str(args.hidden_dim),
        '--embed_dim', str(args.embed_dim),
        '--threshold', str(args.threshold),
        '--embeddings_path', EMBEDDINGS_PATH,
        '--encoders_path', ENCODER_CKPT,
        '--split_path', SPLIT_PATH,
        '--data_root', DATA_ROOT,
        '--device', args.device
    ]
    if args.no_contrastive_aug: cmd.append('--no_aug')
    subprocess.run(cmd, check=True, env=ENV)

def train_generator():
    print(f"\n🚀 STAGE 2: ProM3E Generator ({args.generator_divergence.upper()})")
    cmd = [
        sys.executable, 'model/generator/train_generator.py',
        '--out', GENERATOR_CKPT,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--split_path', SPLIT_PATH,
        '--epochs', str(args.generator_epochs),
        '--lr', str(args.generator_lr),
        '--weight_decay', str(args.generator_weight_decay),
        '--alpha', str(args.generator_alpha),
        '--beta', str(args.generator_beta),
        '--lambd', str(args.generator_lambda),
        '--divergence', args.generator_divergence,
        '--keep_prob', str(args.gen_keep_prob),
        '--kl_warmup', str(args.gen_kl_warmup),
        '--hidden_dim', str(args.gen_hidden_dim),
        '--num_heads', str(args.gen_num_heads),
        '--num_layers', str(args.gen_num_layers),
        '--num_registers', str(args.gen_num_registers),
        '--mlp_depth', str(args.gen_mlp_depth),
        '--dropout', str(args.gen_dropout),
        '--device', args.device
    ]
    subprocess.run(cmd, check=True, env=ENV)

def run_smart_reconstruction():
    print(f"\n🧠 INTERMEDIATE: Smart Hybrid Reconstruction")
    script_path = 'model/generator/run_generator_demo.py'
    cmd = [
        sys.executable, script_path,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--hidden_dim', str(args.gen_hidden_dim),
        '--num_heads', str(args.gen_num_heads),
        '--num_layers', str(args.gen_num_layers),
        '--num_registers', str(args.gen_num_registers),
        '--mlp_depth', str(args.gen_mlp_depth),
        '--device', args.device
    ]
    subprocess.run(cmd, check=True, env=ENV)

def run_downstream():
    print('\n📊 STAGE 3: Granular Downstream Tasks')
    # 1. Classification
    subprocess.run([
        sys.executable, 'model/downstream/downstream_classification.py',
        '--epochs', str(args.cls_epochs),
        '--lr', str(args.downstream_lr),
        '--csv_path', DATA_CSV,
        '--embeddings_path', RECON_DEMO_PATH, 
        '--device', args.device
    ], check=True, env=ENV)

    # 2. Progression
    for idx, name in [(1, 'U2_ADL'), (2, 'U3_Motor')]:
        subprocess.run([
            sys.executable, 'model/downstream/downstream_progression.py',
            '--target_idx', str(idx),
            '--lr', str(args.downstream_lr),
            '--progression_ckpt', os.path.join(CHECKPOINTS_DIR, f'prog_{name}.pt'),
            '--epochs', str(args.prog_epochs),
            '--csv_path', DATA_CSV,
            '--embeddings_path', RECON_DEMO_PATH,
            '--device', args.device
        ], check=True, env=ENV)

    # 3. Static UPDRS
    for idx, name in [(1, 'U2_ADL'), (2, 'U3_Motor')]:
        subprocess.run([
            sys.executable, 'model/downstream/downstream_updrs.py',
            '--target_idx', str(idx),
            '--lr', str(args.downstream_lr),
            '--updrs_ckpt', os.path.join(CHECKPOINTS_DIR, f'static_{name}.pt'),
            '--epochs', str(args.updrs_epochs),
            '--csv_path', DATA_CSV,
            '--embeddings_path', RECON_DEMO_PATH,
            '--device', args.device
        ], check=True, env=ENV)

if __name__ == '__main__':
    run_contrastive()
    train_generator()
    run_smart_reconstruction() 
    run_downstream()
    print('\n✅ Full ProM3E Pipeline Complete.')