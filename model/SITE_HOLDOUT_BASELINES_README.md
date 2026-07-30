# Unified site-held-out raw-graph baselines

`run_site_holdout_baselines.py` trains every baseline and downstream task in one
resumable invocation using the exact splits from a completed
`run_5fold_site_cv.py` experiment.

The models are:

- SPECT: GCN encoder;
- MRI: GCN encoder;
- fMRI: GINE encoder;
- DTI: GINE encoder;
- naïve multimodal: all four modality-specific encoders, concatenation, zero
  filling for unavailable modality slots, and an explicit availability mask.

All encoders are trained end-to-end from raw graphs for each downstream task.
The baselines do not use contrastive alignment or generative reconstruction.
The full-model row is read from the existing fold checkpoints, ensuring the
same untouched test sites.

Three-class classification is reported only if Control, PD, and Prodromal are
represented in training, validation, and test for every outer fold. Therefore,
modalities with no observed Prodromal scans are marked `--`; the script never
silently converts a three-class task into a binary task.

## Background run

```bash
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  .venv-uv/bin/python model/run_site_holdout_baselines.py \
  --site_cv_root model/logs/site_5fold_cv/site_5fold_seed42_20260724_170258 \
  --data_root data \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --output_dir model/results/site_holdout_baselines_seed42 \
  --device cuda \
  > site_holdout_baselines.log 2>&1 < /dev/null &

echo $!
tail -f site_holdout_baselines.log
```

There are 125 baseline jobs by default: five baselines, five downstream tasks,
and five outer folds. Each completed job writes a JSON result immediately.
Restarting the same command automatically skips completed jobs. Use `--restart`
only to intentionally retrain all requested jobs.

## Outputs

- `runs/fold_N/BASELINE/TASK.json`: resumable per-job result and cohort counts;
- `per_fold_metrics.csv`: untouched test metrics for every evaluable job;
- `site_holdout_baseline_summary.csv`: fold mean, sample SD, and fold values;
- `site_holdout_baseline_table.tex`: complete manuscript table;
- `run_manifest.json`: split audits and exact hyperparameters.

The generated table uses mean ± sample standard deviation across the five outer
test folds and bolds the best valid mean in each column. It includes the
existing full-model row from the same site-held-out experiment.
