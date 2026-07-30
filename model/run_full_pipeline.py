#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
import json
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline with Smart Hybrid Fusion.")
    
    # --- Paths & General ---
    parser.add_argument('--data_csv', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'PPMI_Curated_Data_Cut_Public_20251112.csv')))
    parser.add_argument('--split_path', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_split_master.txt')))
    parser.add_argument('--checkpoints_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints')))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--reuse_representations_dir',
        help=(
            'Skip Stage 1/2 and reuse recon_demo.pt (or embeddings.pt with '
            '--disable_generator) from this directory for a matched downstream rerun.'
        ),
    )
    
    # --- Ablation Flags ---
    parser.add_argument('--exclude_modality', nargs='+', default=[], choices=['SPECT', 'MRI', 'fMRI', 'DTI'])
    parser.add_argument('--skip_cl', action='store_true')
    parser.add_argument('--disable_generator', action='store_true')
    parser.add_argument(
        '--no_missingness_mask',
        action='store_true',
        help='Train all downstream heads without the observed/reconstructed modality indicators.',
    )
    parser.add_argument('--drop_prodromal', action='store_true')
    
    # 🌟 NEW: Autonomous Ablation Routing Flags
    parser.add_argument('--ablate_modality', type=str, choices=['fMRI', 'DTI', 'MRI', 'SPECT'], help="Modality to partially mask out downstream")
    parser.add_argument('--ablate_ratio', type=float, help="Percentage to KEEP (e.g., 0.1 for 10%)")
    
    # --- Strict Mode Flags ---
    parser.add_argument('--require_all_active', action='store_true', help="Only train/eval on subjects possessing ALL active modalities")
    parser.add_argument('--strict_downstream', action='store_true', help="Use ALL data for Stage 1/2, but STRICT data for Stage 3")
    
    # --- Stage 1: Contrastive Pre-training ---
    parser.add_argument('--contrastive_epochs', type=int, default=100)
    parser.add_argument('--contrastive_lr', type=float, default=0.00015) 
    parser.add_argument('--contrastive_batch_size', type=int, default=32) 
    parser.add_argument('--contrastive_alpha', type=float, default=-8.66) 
    parser.add_argument('--contrastive_beta', type=float, default=8.32)   
    parser.add_argument('--hub_name', '--hub_modality', default='fMRI', choices=['fMRI', 'MRI', 'SPECT', 'DTI'])
    parser.add_argument('--alternate_hub', default=None, choices=['fMRI', 'MRI', 'SPECT', 'DTI'],
                        help="If the requested hub is excluded, use this hub instead")
    parser.add_argument('--aug_mask', type=float, default=0.24)           
    parser.add_argument('--aug_jitter', type=float, default=0.01)         
    parser.add_argument('--clip_val', type=float, default=2.3)            
    parser.add_argument('--hidden_dim', type=int, default=256)            
    parser.add_argument('--embed_dim', type=int, default=1024)            
    parser.add_argument('--threshold', type=float, default=0.60)         
    parser.add_argument('--no_contrastive_aug', action='store_true')
    
    # --- Stage 2: Generator ---
    parser.add_argument('--generator_epochs', type=int, default=100)
    parser.add_argument('--generator_lr', type=float, default=0.000012)     
    parser.add_argument('--generator_weight_decay', type=float, default=0.004) 
    parser.add_argument('--generator_alpha', type=float, default=-4.41)     
    parser.add_argument('--generator_beta', type=float, default=1.41)       
    parser.add_argument('--generator_lambda', type=float, default=0.00043)  
    parser.add_argument('--generator_divergence', choices=['mmd', 'kl', 'none'], default='kl') 
    parser.add_argument('--gen_keep_prob', type=float, default=0.5)         
    parser.add_argument('--gen_kl_warmup', type=int, default=12)           
    parser.add_argument('--gen_hidden_dim', type=int, default=1024)         
    parser.add_argument('--gen_num_heads', type=int, default=8)             
    parser.add_argument('--gen_num_layers', type=int, default=5)            
    parser.add_argument('--gen_num_registers', type=int, default=0)         
    parser.add_argument('--gen_mlp_depth', type=int, default=3)             
    parser.add_argument('--gen_dropout', type=float, default=0.28)          
    
    # --- Stage 3: Downstream Tasks ---
    parser.add_argument('--cls_only', action='store_true', help="Only run classification, skip progression and UPDRS.")
    parser.add_argument('--downstream_lr', type=float, default=0.01)        
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
GENERATOR_CKPT = os.path.join(CHECKPOINTS_DIR, 'generator.pt')
RECON_DEMO_PATH = os.path.join(CHECKPOINTS_DIR, 'recon_demo.pt')

ENV = os.environ.copy()
ENV['PYTHONPATH'] = '.'

STAGE_TIMES = {}


def _format_seconds(seconds):
    minutes, secs = divmod(float(seconds), 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:05.2f}s"
    return f"{secs:.2f}s"


def _record_timing(stage_name, elapsed_seconds):
    STAGE_TIMES[stage_name] = float(elapsed_seconds)
    print(f"⏱️  {stage_name} took {_format_seconds(elapsed_seconds)}")


def _timed(stage_name, fn, *fn_args, **fn_kwargs):
    t0 = time.perf_counter()
    out = fn(*fn_args, **fn_kwargs)
    elapsed = time.perf_counter() - t0
    _record_timing(stage_name, elapsed)
    return out


def save_timing_report():
    if not STAGE_TIMES:
        return

    total = sum(STAGE_TIMES.values())
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'timings_seconds': STAGE_TIMES,
        'total_seconds': total
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    latest_path = os.path.join(CHECKPOINTS_DIR, 'pipeline_timing_latest.json')
    run_path = os.path.join(CHECKPOINTS_DIR, f'pipeline_timing_{ts}.json')

    for path in [latest_path, run_path]:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    print("\n📊 Pipeline Timing Summary")
    for name, seconds in STAGE_TIMES.items():
        print(f"  - {name}: {_format_seconds(seconds)}")
    print(f"  - TOTAL: {_format_seconds(total)}")
    print(f"\n📝 Saved timing report to: {latest_path}")
    print(f"📝 Saved run-specific report to: {run_path}")

# Intelligent Split Management
DOWNSTREAM_SPLIT_PATH = SPLIT_PATH

if args.require_all_active or args.strict_downstream:
    all_mods = ['SPECT', 'MRI', 'fMRI', 'DTI']
    exclude_list = args.exclude_modality if args.exclude_modality else []
    active_mods = [m for m in all_mods if m not in exclude_list]
    
    print(f"\n🔍 STRICT MODE: Filtering {os.path.basename(SPLIT_PATH)} to only include subjects possessing: {active_mods}")
    
    strict_split_path = os.path.join(CHECKPOINTS_DIR, 'strict_split.txt')
    kept_count, total_count = 0, 0
    
    with open(SPLIT_PATH, 'r') as f:
        lines = f.read().splitlines()
        
    with open(strict_split_path, 'w') as f:
        for line in lines:
            if any(header in line for header in ('train_ids:', 'val_ids:', 'test_ids:')) or not line.strip() or line.startswith('#'):
                f.write(line + '\n')
            else:
                subj = line.strip()
                total_count += 1
                has_all = all(os.path.exists(os.path.join(DATA_ROOT, mod, f"{subj}.pt")) for mod in active_mods)
                if has_all:
                    f.write(subj + '\n')
                    kept_count += 1
                    
    print(f"🎯 Filtered down to {kept_count}/{total_count} strict subjects.")
    
    if args.require_all_active:
        SPLIT_PATH = strict_split_path
        DOWNSTREAM_SPLIT_PATH = strict_split_path
        print("⚠️ Applied Strict Split GLOBALLY (All Stages).")
    elif args.strict_downstream:
        DOWNSTREAM_SPLIT_PATH = strict_split_path
        print("⚠️ Applied Strict Split to DOWNSTREAM ONLY. Stage 1 will use full data.")

def build_cmd(base_cmd):
    if args.exclude_modality:
        base_cmd.append('--exclude_modality')
        base_cmd.extend(args.exclude_modality)
    if args.disable_generator:
        base_cmd.append('--disable_generator')
    if args.no_missingness_mask:
        base_cmd.append('--no_missingness_mask')
    return base_cmd

def run_contrastive():
    print(f"\n STAGE 1: Contrastive Alignment (Hub: {args.hub_name})")

    # Handle configuration where the requested hub is excluded.
    if args.exclude_modality and args.hub_name in args.exclude_modality:
        # Prefer explicit alternate hub if provided and valid
        if args.alternate_hub and args.alternate_hub not in args.exclude_modality:
            print(f"ℹ️ Requested hub '{args.hub_name}' excluded; switching to alternate hub '{args.alternate_hub}'.")
            args.hub_name = args.alternate_hub
        else:
            # Auto-select the first available modality not in exclude_modality
            available = [m for m in ['fMRI', 'MRI', 'SPECT', 'DTI'] if m not in args.exclude_modality]
            if available:
                new_hub = available[0]
                print(f"ℹ️ Requested hub '{args.hub_name}' excluded; auto-selecting hub '{new_hub}'.")
                args.hub_name = new_hub
            else:
                print("❌ No available hub modalities left after applying --exclude_modality.")
                sys.exit(2)

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
    if args.exclude_modality:
        cmd.append('--exclude_modality')
        cmd.extend(args.exclude_modality)
    subprocess.run(cmd, check=True, env=ENV)

def train_generator():
    print(f"\n STAGE 2: Generator ({args.generator_divergence.upper()})")
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
    if args.exclude_modality:
        cmd.append('--exclude_modality')
        cmd.extend(args.exclude_modality)
    subprocess.run(cmd, check=True, env=ENV)

def run_smart_reconstruction():
    print(f"\n INTERMEDIATE: Smart Hybrid Reconstruction")
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
    if args.exclude_modality:
        cmd.append('--exclude_modality')
        cmd.extend(args.exclude_modality)
    subprocess.run(cmd, check=True, env=ENV)

def run_downstream(target_embeddings_path):
    print('\n STAGE 3: Granular Downstream Tasks')
    
    # 1. Classification
    t_cls = time.perf_counter()
    cmd_cls_list = [
        sys.executable, 'model/downstream/downstream_classification.py',
        '--epochs', str(args.cls_epochs),
        '--lr', str(args.downstream_lr),
        '--csv_path', DATA_CSV,
        '--embeddings_path', target_embeddings_path, 
        '--classifier_ckpt', os.path.join(CHECKPOINTS_DIR, 'classifier.pt'),
        '--split_path', DOWNSTREAM_SPLIT_PATH, 
        '--device', args.device,
        '--seed', str(args.seed),
    ]
    
    if getattr(args, 'drop_prodromal', False) or getattr(args, 'strict_downstream', False) or getattr(args, 'require_all_active', False):
        cmd_cls_list.append('--drop_prodromal')

    # 🌟 THIS IS THE CRITICAL ADDITION 🌟
    # Safely passes the new variables strictly to the classification script
    if args.ablate_modality and args.ablate_ratio is not None:
        cmd_cls_list.extend(['--ablate_modality', args.ablate_modality])
        cmd_cls_list.extend(['--ablate_ratio', str(args.ablate_ratio)])
        
    cmd_cls = build_cmd(cmd_cls_list)
    subprocess.run(cmd_cls, check=True, env=ENV)
    _record_timing('Stage 3.1 - Classification', time.perf_counter() - t_cls)

    # 🛑 EARLY EXIT IF CLS_ONLY IS ACTIVE
    if args.cls_only:
        print("\n⚠️ Skipping Progression and Static UPDRS tasks (--cls_only active).")
        return

    # 2. Progression
    t_prog = time.perf_counter()
    for idx, name in [(1, 'U2_ADL'), (2, 'U3_Motor')]:
        cmd_prog = build_cmd([
            sys.executable, 'model/downstream/downstream_progression.py',
            '--target_idx', str(idx),
            '--lr', str(args.downstream_lr),
            '--progression_ckpt', os.path.join(CHECKPOINTS_DIR, f'prog_{name}.pt'),
            '--epochs', str(args.prog_epochs),
            '--csv_path', DATA_CSV,
            '--hidden_dim', str(args.hidden_dim),
            '--embeddings_path', target_embeddings_path,
            '--split_path', DOWNSTREAM_SPLIT_PATH, 
            '--device', args.device,
            '--seed', str(args.seed),
        ])
        subprocess.run(cmd_prog, check=True, env=ENV)
    _record_timing('Stage 3.2 - Progression (both targets)', time.perf_counter() - t_prog)

    # 3. Static UPDRS
    t_updrs = time.perf_counter()
    for idx, name in [(1, 'U2_ADL'), (2, 'U3_Motor')]:
        cmd_stat = build_cmd([
            sys.executable, 'model/downstream/downstream_updrs.py',
            '--target_idx', str(idx),
            '--lr', str(args.downstream_lr),
            '--updrs_ckpt', os.path.join(CHECKPOINTS_DIR, f'static_{name}.pt'),
            '--epochs', str(args.updrs_epochs),
            '--csv_path', DATA_CSV,
            '--split_path', DOWNSTREAM_SPLIT_PATH, 
            '--embeddings_path', target_embeddings_path,
            '--device', args.device,
            '--seed', str(args.seed),
        ])
        subprocess.run(cmd_stat, check=True, env=ENV)
    _record_timing('Stage 3.3 - Static UPDRS (both targets)', time.perf_counter() - t_updrs)

if __name__ == '__main__':
    pipeline_t0 = time.perf_counter()

    if args.reuse_representations_dir:
        reuse_dir = os.path.abspath(args.reuse_representations_dir)
        reuse_filename = 'embeddings.pt' if args.disable_generator else 'recon_demo.pt'
        embeddings_to_use = os.path.join(reuse_dir, reuse_filename)
        if not os.path.exists(embeddings_to_use):
            raise FileNotFoundError(
                f"Cannot reuse representations; required artifact is missing: {embeddings_to_use}"
            )
        print(f"\n♻️ Reusing fixed downstream representations: {embeddings_to_use}")
        _record_timing('Stage 1 - Contrastive Alignment', 0.0)
        _record_timing('Stage 2 - Generator Training', 0.0)
        _record_timing('Stage 2.5 - Smart Reconstruction Demo', 0.0)
    else:
        if args.skip_cl:
            print("\n⚠️ ABLATION: Skipping Contrastive Learning")
            args.contrastive_epochs = 0

        _timed('Stage 1 - Contrastive Alignment', run_contrastive)

        if args.disable_generator:
            print("\n⚠️ ABLATION: Disabling Generative Reconstruction")
            embeddings_to_use = EMBEDDINGS_PATH
            _record_timing('Stage 2 - Generator Training', 0.0)
            _record_timing('Stage 2.5 - Smart Reconstruction Demo', 0.0)
        else:
            _timed('Stage 2 - Generator Training', train_generator)
            _timed('Stage 2.5 - Smart Reconstruction Demo', run_smart_reconstruction)
            embeddings_to_use = RECON_DEMO_PATH
        
    _timed('Stage 3 - Downstream Tasks (total)', run_downstream, embeddings_to_use)
    _record_timing('Pipeline - End to End', time.perf_counter() - pipeline_t0)
    save_timing_report()
    print('\n Full GeNeuro Pipeline Complete.')
