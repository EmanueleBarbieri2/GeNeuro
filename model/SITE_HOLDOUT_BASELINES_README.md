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
  filling for unavailable modality slots, and no explicit availability mask.

All encoders are trained end-to-end from raw graphs for each downstream task.
The baselines do not use contrastive alignment or generative reconstruction.
The full-model row is read from maskless fold checkpoints. Before training,
the runner verifies that its train/validation/test IDs exactly match the raw
baseline fold files and that every downstream checkpoint records
`use_missingness_mask=False`. For every fold and task, it also requires the
naïve multimodal and full model to have identical test sample counts and
SHA-256 hashes.

MRI, fMRI, naïve multimodal, and full-model classification use Control, PD,
and Prodromal. Because SPECT and DTI contain no observed Prodromal scans, those
two rows use a clearly marked binary Control-versus-PD endpoint. Binary and
three-class classification values share the table but are not ranked against
one another. MRI progression remains unavailable: a raw MRI longitudinal
baseline cannot be defined because no patient has two observed MRI visits.

## Background run

```bash
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  .venv-uv/bin/python model/run_site_holdout_baselines.py \
  --site_cv_root model/logs/site_5fold_cv/site_5fold_seed42_20260724_170258 \
  --full_model_root model/logs/site_5fold_cv/MASKLESS_FULL_MODEL_RUN \
  --data_root data \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --output_dir model/results/site_holdout_baselines_maskless_seed42 \
  --device cuda \
  --no-multimodal_missingness_mask \
  > site_holdout_baselines_maskless.log 2>&1 < /dev/null &

echo $!
tail -f site_holdout_baselines_maskless.log
```

There are 125 baseline jobs by default: five baselines, five downstream tasks,
and five outer folds. Each completed job writes a JSON result immediately.
Restarting the same command automatically skips completed jobs. Use `--restart`
only to intentionally retrain all requested jobs.

## Outputs

- `runs/fold_N/BASELINE/TASK.json`: resumable per-job result and cohort counts;
- `per_fold_metrics.csv`: untouched test metrics for every evaluable job;
- `multimodal_full_comparability_audit.csv`: exact test sample/hash audit;
- `availability_only_diagnostics.csv`: train-pattern-only prediction on each
  untouched site test fold, quantifying the missingness shortcut;
- `availability_only_summary.csv`: mean and sample SD of that diagnostic;
- `gcn_gine_vs_full_diagnostic.md`: matched-sample performance differences and
  evidence-based interpretation;
- `site_holdout_baseline_summary.csv`: fold mean, sample SD, and fold values;
- `site_holdout_baseline_table.tex`: complete manuscript table;
- `run_manifest.json`: split audits and exact hyperparameters.

The generated table uses mean ± sample standard deviation across the five outer
test folds and bolds the best valid mean in each column. Binary SPECT/DTI
classification is excluded from three-class best-value selection.

Zero-filled modality blocks still reveal which modalities are absent even
without the four explicit flags. The availability-only diagnostic is therefore
required when interpreting a strong naïve multimodal result.
