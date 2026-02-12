#!/usr/bin/env python3
"""
Automate the full pipeline: generator, embeddings, downstream tasks, and explainers.
Assumes working directory is model/model and data is in ../data/PPMI_Curated_Data_Cut_Public_20251112.csv
"""

import subprocess
import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline for contrastive, generator, downstream, and explainers.")
    # General
    parser.add_argument('--data_csv', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'PPMI_Curated_Data_Cut_Public_20251112.csv')))
    parser.add_argument('--split_path', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'unified_split.txt')))
    parser.add_argument('--checkpoints_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints')))
    parser.add_argument('--device', default='cuda' if os.path.exists('/dev/nvidia0') else 'cpu')
    
    # Training Hyperparameters
    parser.add_argument('--contrastive_epochs', type=int, default=100)
    parser.add_argument('--contrastive_batch_size', type=int, default=32)
    parser.add_argument('--contrastive_lr', type=float, default=5e-4)
    
    parser.add_argument('--generator_epochs', type=int, default=30)
    parser.add_argument('--generator_lr', type=float, default=1e-3)
    
    parser.add_argument('--downstream_epochs', type=int, default=30)
    parser.add_argument('--downstream_lr', type=float, default=1e-3)
    
    # Explainers
    parser.add_argument('--explainer_topk', type=int, default=10)
    parser.add_argument('--explainer_delta_t', type=float, default=1.0)
    parser.add_argument('--run_explainers', action='store_true', default=False, help='Run explainers step (default: skip)')
    
    return parser.parse_args()


args = parse_args()

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = args.checkpoints_dir
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DATA_CSV = args.data_csv
SPLIT_PATH = args.split_path
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

# Checkpoint Files
EMBEDDINGS_PATH = os.path.join(CHECKPOINTS_DIR, 'embeddings.pt')
ENCODER_CKPT = os.path.join(CHECKPOINTS_DIR, 'encoders.pt')
GENERATOR_CKPT = os.path.join(CHECKPOINTS_DIR, 'prom3e_generator.pt')
PROGRESSION_CKPT = os.path.join(CHECKPOINTS_DIR, 'progression_regressor.pt')
UPDRS_CKPT = os.path.join(CHECKPOINTS_DIR, 'updrs_regressor.pt')
CLASSIFIER_CKPT = os.path.join(CHECKPOINTS_DIR, 'classifier.pt')
RECON_DEMO_PATH = os.path.join(CHECKPOINTS_DIR, 'recon_demo.pt') # Separate file for generated embeddings

def run_contrastive():
    print(f"\n--- Running Contrastive Pre-training (Output: {EMBEDDINGS_PATH}) ---")
    cmd = [
        sys.executable, 'model/contrastive/train.py',
        '--epochs', str(args.contrastive_epochs),
        '--batch_size', str(args.contrastive_batch_size),
        '--lr', str(args.contrastive_lr),
        '--out_encoders', ENCODER_CKPT,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--encoders_path', ENCODER_CKPT,
        '--split_path', SPLIT_PATH,  # <--- INJECTED SPLIT PATH
        '--data_root', DATA_ROOT,
        '--device', args.device
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    subprocess.run(cmd, check=True, env=env)

def train_generator():
    print(f"\n--- Training Generator (Output: {GENERATOR_CKPT}) ---")
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    subprocess.run([
        sys.executable, 'model/generator/train_generator.py',
        '--csv_path', DATA_CSV,
        '--out', GENERATOR_CKPT,
        '--embeddings_path', EMBEDDINGS_PATH, # Uses REAL embeddings
        '--split_path', SPLIT_PATH,            # <--- INJECTED SPLIT PATH
        '--epochs', str(args.generator_epochs),
        '--lr', str(args.generator_lr),
        '--device', args.device
    ], check=True, env=env)

def generate_embeddings():
    print(f"\n--- Generating Demo Reconstructions (Output: {RECON_DEMO_PATH}) ---")
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    # IMPORTANT: We verify that this script saves to a distinct file (e.g., recon_demo.pt)
    # inside run_generator_demo.py, or we pass an output arg if supported.
    # Assuming run_generator_demo.py takes input args:
    subprocess.run([
        sys.executable, 'model/generator/run_generator_demo.py', 
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--out', RECON_DEMO_PATH, # Ensure separate output
        '--device', args.device
    ], check=True, env=env)

def run_downstream():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    print('\n--- Downstream Classification ---')
    subprocess.run([
        sys.executable, 'model/downstream/downstream_classification.py',
        '--epochs', str(args.downstream_epochs),
        '--lr', str(args.downstream_lr),
        '--device', args.device,
        '--csv_path', DATA_CSV,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT
    ], check=True, env=env)
    
    print('\n--- Downstream Progression ---')
    subprocess.run([
        sys.executable, 'model/downstream/downstream_progression.py',
        '--epochs', str(args.downstream_epochs),
        '--lr', str(args.downstream_lr),
        '--device', args.device,
        '--csv_path', DATA_CSV,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--progression_ckpt', PROGRESSION_CKPT
    ], check=True, env=env)
    
    print('\n--- Downstream UPDRS ---')
    subprocess.run([
        sys.executable, 'model/downstream/downstream_updrs.py',
        '--epochs', str(args.downstream_epochs),
        '--lr', str(args.downstream_lr),
        '--device', args.device,
        '--csv_path', DATA_CSV,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--updrs_ckpt', UPDRS_CKPT
    ], check=True, env=env)

def run_explainers():
    print('\n--- Running Explainers ---')
    env = os.environ.copy()
    # Set PYTHONPATH to root so 'model' is importable
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env['PYTHONPATH'] = workspace_root
    
    subprocess.run([
        sys.executable, 'model/xplainers/batch_explainers.py',
        '--data_root', DATA_ROOT,
        '--atlas', os.path.abspath(os.path.join(workspace_root, 'atlas_centroids.csv')),
        '--lut', os.path.abspath(os.path.join(workspace_root, 'FreeSurferColorLUT.txt')),
        '--out_dir', os.path.abspath(os.path.join(workspace_root, 'xplainers', 'explainer_reports')),
        '--encoder_ckpt', ENCODER_CKPT,
        '--generator_ckpt', GENERATOR_CKPT,
        '--progression_ckpt', PROGRESSION_CKPT,
        '--updrs_ckpt', UPDRS_CKPT,
        '--topk', str(args.explainer_topk),
        '--delta_t', str(args.explainer_delta_t),
        '--device', args.device
    ], check=True, env=env, cwd=workspace_root)

    # Classification explainers (per subject)
    print('Generating classification explainer reports...')
    import csv
    class_report_dir = os.path.join(workspace_root, 'xplainers', 'explainer_reports', 'classification')
    os.makedirs(class_report_dir, exist_ok=True)
    
    # Get all subject_visit IDs from the CSV
    subject_visits = []
    with open(DATA_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patno = row.get('PATNO')
            event_id = row.get('EVENT_ID')
            if patno and event_id:
                subject_visits.append(f"{patno}_{event_id}")
                
    subject_visits = sorted(set(subject_visits))
    
    for sid in subject_visits:
        out_json = os.path.join(class_report_dir, f'{sid}.json')
        out_txt = os.path.join(class_report_dir, f'{sid}.txt')
        
        # Run raw explainer
        # Note: explain_classification.py needs to run from within model/xplainers context or with proper path
        subprocess.run([
            sys.executable, 'model/xplainers/explain_classification.py',
            '--subject', sid,
            '--csv_path', DATA_CSV,
            '--embeddings_path', EMBEDDINGS_PATH,
            '--generator_ckpt', GENERATOR_CKPT,
            '--classifier_ckpt', CLASSIFIER_CKPT,
            '--out', out_json
        ], check=True, env=env) # Pass env with PYTHONPATH
        
        # Run readable explainer
        with open(out_txt, 'w') as repf:
            subprocess.run([
                sys.executable, 'model/xplainers/explain_classification_readable.py',
                '--infile', out_json
            ], check=True, stdout=repf, stderr=subprocess.STDOUT, env=env)

if __name__ == '__main__':
    run_contrastive()
    train_generator()
    generate_embeddings() # Saves to separate file (recon_demo.pt)
    run_downstream()
    
    if args.run_explainers:
        run_explainers()
    else:
        print('Skipping explainers step (default).')
        
    print('\n✅ Full pipeline complete.')