MINERVA: Multimodal Integration of Neuroimaging via Embedding Reconstruction and Variational Alignment
===============================================================

Overview
--------
This repository contains an end-to-end research pipeline for learning multimodal brain representations from heterogeneous neuroimaging modalities (structural MRI, functional MRI, DTI, and SPECT) and applying those representations to downstream clinical prediction tasks. The project implements a three-stage workflow:

- Stage 1 — Contrastive alignment: modality-specific encoders are trained to align modality embeddings into a shared space.
- Stage 2 — Generative reconstruction: a generative model learns to reconstruct missing modalities from embeddings.
- Stage 3 — Granular downstream tasks: classification, longitudinal progression modeling, and static clinical score (UPDRS) prediction using learned or reconstructed embeddings.

The codebase is organized to run either a single end-to-end pipeline or a 5-fold cross-validation orchestration.


Important files
---------------

- Orchestration & entry points:
  - [model/run_full_pipeline.py](model/run_full_pipeline.py) — run the full 3-stage pipeline for a given split and checkpoint folder.
  - [model/run_5fold_cv.py](model/run_5fold_cv.py) — run the full pipeline across 5 folds and aggregate metrics.

- Core model components:
  - [model/encoders.py](model/encoders.py) — modality-specific encoder implementations (fMRI, DTI, MRI, SPECT).
  - [model/contrastive/train.py](model/contrastive/train.py) — contrastive pre-training script that exports embeddings and encoder weights.
  - [model/generator/generator.py](model/generator/generator.py) — generator architecture used for modality reconstruction.
  - [model/generator/train_generator.py](model/generator/train_generator.py) — training script for the generator (used by the pipeline).

- Downstream code & explainers:
  - [model/downstream/downstream_classification.py](model/downstream/downstream_classification.py)
  - [model/downstream/downstream_progression.py](model/downstream/downstream_progression.py)
  - [model/downstream/downstream_updrs.py](model/downstream/downstream_updrs.py)
  - [model/xplainers](model/xplainers) — explainability utilities and global/local biomarker aggregation.

- Data & splits:
  - [data/PPMI_Curated_Data_Cut_Public_20251112.csv](data/PPMI_Curated_Data_Cut_Public_20251112.csv) — curated metadata table used by downstream scripts.
  - [data/unified_split_*.txt](data/) — train/val split files for CV and the master split.

Dependencies
------------
Install the project's Python dependencies in a virtual environment. The project includes a `requirements.txt` file at the repository root.

Quickstart
----------
1. Create and activate a Python environment (recommended: conda or venv) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the full pipeline for one split (example):

```bash
python model/run_full_pipeline.py \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --split_path data/unified_split_master.txt \
  --checkpoints_dir model/checkpoints/run_demo \
  --device cuda
```

3. Run 5-fold cross-validation and save aggregated results:

```bash
python model/run_5fold_cv.py --device cuda --logs_dir model/logs/cv_demo
```

