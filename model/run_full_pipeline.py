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
    parser.add_argument('--checkpoints_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints')))
    parser.add_argument('--device', default='cpu')
    # Contrastive
    parser.add_argument('--contrastive_epochs', type=int, default=50)
    parser.add_argument('--contrastive_batch_size', type=int, default=8)
    parser.add_argument('--contrastive_lr', type=float, default=1e-4)
    # Generator
    parser.add_argument('--generator_epochs', type=int, default=30)
    parser.add_argument('--generator_lr', type=float, default=1e-3)
    # Downstream classification
    parser.add_argument('--clf_epochs', type=int, default=30)
    parser.add_argument('--clf_lr', type=float, default=1e-3)
    # Downstream progression
    parser.add_argument('--progression_epochs', type=int, default=30)
    parser.add_argument('--progression_lr', type=float, default=1e-3)
    # Downstream updrs
    parser.add_argument('--updrs_epochs', type=int, default=30)
    parser.add_argument('--updrs_lr', type=float, default=1e-3)
    # Explainers
    parser.add_argument('--explainer_topk', type=int, default=10)
    parser.add_argument('--explainer_delta_t', type=float, default=1.0)
    parser.add_argument('--run_explainers', action='store_true', default=False, help='Run explainers step (default: skip)')
    return parser.parse_args()


args = parse_args()
import os

# Always use absolute path for checkpoints in model/checkpoints
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
DATA_CSV = os.path.abspath(args.data_csv)
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
GENERATOR_CKPT = os.path.join(CHECKPOINTS_DIR, 'prom3e_generator.pt')
EMBEDDINGS_PATH = os.path.join(CHECKPOINTS_DIR, 'embeddings.pt')
ENCODER_CKPT = os.path.join(CHECKPOINTS_DIR, 'encoders.pt')
PROGRESSION_REGRESSOR_CKPT = os.path.join(CHECKPOINTS_DIR, 'progression_regressor.pt')

def run_contrastive():
    cmd = [
        sys.executable, 'model/contrastive/train.py',
        '--epochs', str(args.contrastive_epochs),
        '--batch_size', str(args.contrastive_batch_size),
        '--lr', str(args.contrastive_lr),
        '--out_encoders', ENCODER_CKPT,
        '--out_embeddings', EMBEDDINGS_PATH,
        '--data_root', DATA_ROOT,
        '--device', args.device,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--encoders_path', ENCODER_CKPT
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    subprocess.run(cmd, check=True, env=env)
    print('Contrastive training complete. Encoders and embeddings saved to checkpoints/.')

def train_generator():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    subprocess.run([
        sys.executable, 'model/generator/train_generator.py',
        '--csv_path', DATA_CSV,
        '--out', GENERATOR_CKPT,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--epochs', str(args.generator_epochs),
        '--lr', str(args.generator_lr),
        '--device', args.device
    ], check=True, env=env)
    print('Training generator...')

def generate_embeddings():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    # Only save reconstructions to recon_demo.pt, do not overwrite embeddings.pt
    subprocess.run([
        sys.executable, 'model/generator/run_generator_demo.py',
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--device', args.device
    ], check=True, env=env)
    print('Generating reconstructions...')

def run_downstream():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    subprocess.run([
        sys.executable, 'model/downstream/downstream_classification.py',
        '--epochs', str(args.clf_epochs),
        '--lr', str(args.clf_lr),
        '--device', args.device,
        '--csv_path', DATA_CSV,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT
    ], check=True, env=env)
    print('Running downstream classification...')
    subprocess.run([
        sys.executable, 'model/downstream/downstream_progression.py',
        '--epochs', str(args.progression_epochs),
        '--lr', str(args.progression_lr),
        '--device', args.device,
        '--csv_path', DATA_CSV,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--progression_ckpt', PROGRESSION_REGRESSOR_CKPT
    ], check=True, env=env)
    print('Running downstream progression...')
    subprocess.run([
        sys.executable, 'model/downstream/downstream_updrs.py',
        '--epochs', str(args.updrs_epochs),
        '--lr', str(args.updrs_lr),
        '--device', args.device,
        '--csv_path', DATA_CSV,
        '--embeddings_path', EMBEDDINGS_PATH,
        '--generator_ckpt', GENERATOR_CKPT,
        '--updrs_ckpt', os.path.join(CHECKPOINTS_DIR, 'updrs_regressor.pt')
    ], check=True, env=env)
    print('Running downstream updrs...')

def run_explainers():
    env = os.environ.copy()
    # Set PYTHONPATH and cwd to workspace root so 'model' is importable
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
        '--progression_ckpt', PROGRESSION_REGRESSOR_CKPT,
        '--updrs_ckpt', os.path.join(CHECKPOINTS_DIR, 'updrs_regressor.pt'),
        '--topk', str(args.explainer_topk),
        '--delta_t', str(args.explainer_delta_t),
        '--device', args.device
    ], check=True, env=env, cwd=workspace_root)

    # 2. Classification explainer reports (per subject)
    print('Generating classification explainer reports...')
    import csv
    class_report_dir = os.path.join('xplainers', 'explainer_reports', 'classification')
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
        subprocess.run([
            sys.executable, 'model/xplainers/explain_classification.py',
            '--subject', sid,
            '--csv_path', DATA_CSV,
            '--embeddings_path', os.path.join(CHECKPOINTS_DIR, 'embeddings.pt'),
            '--generator_ckpt', GENERATOR_CKPT,
            '--classifier_ckpt', os.path.join(CHECKPOINTS_DIR, 'classifier.pt'),
            '--out', out_json
        ], check=True)
        # Run readable explainer
        with open(out_txt, 'w') as repf:
            subprocess.run([
                sys.executable, 'model/xplainers/explain_classification_readable.py',
                '--infile', out_json
            ], check=True, stdout=repf, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    run_contrastive()
    train_generator()
    generate_embeddings()
    run_downstream()
    if args.run_explainers:
        run_explainers()
    else:
        print('Skipping explainers step (default).')
    print('Full pipeline complete.')
