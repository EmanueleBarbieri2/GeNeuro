# Fresh site-held-out end-to-end benchmark

Use `run_site_holdout_end_to_end_benchmark.py` for the manuscript comparison.
It runs two independent branches on the same patient- and site-disjoint folds:

- **Full model:** newly trained contrastive encoders, generator,
  reconstructions, and downstream heads.
- **GCN/GINE baselines:** newly initialized modality-specific encoders trained
  directly from raw graph inputs for each supervised task.

The baseline branch does not load `encoders.pt`, `embeddings.pt`,
`generator.pt`, `recon_demo.pt`, or any full-model downstream checkpoint.
Every fold × baseline × task creates a new encoder and downstream head, so
even two baseline tasks do not share learned weights or optimizer state. The
only state restored is the best validation epoch from that exact same training
run immediately before its untouched-test evaluation. Availability indicators
are disabled in both branches.

Run on GPU 1:

```bash
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  .venv-uv/bin/python model/run_site_holdout_end_to_end_benchmark.py \
  --site_cv_root model/logs/site_5fold_cv/site_5fold_seed42_20260724_170258 \
  --data_root data \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --output_dir model/results/site_holdout_end_to_end_seed42 \
  --device cuda \
  > site_holdout_end_to_end.log 2>&1 < /dev/null &

echo $!
```

Monitor:

```bash
tail -f site_holdout_end_to_end.log
```

The final manuscript table is written to:

```text
model/results/site_holdout_end_to_end_seed42/site_holdout_end_to_end_table.tex
```

Use a new output directory for a completely clean experiment. On later
invocations the script resumes only its own outputs. Pass `--restart_full`
and/or `--restart_baselines` to explicitly retrain a branch.
