# Legacy representation-level site-held-out experiment

> This is an ablation/legacy-reproduction utility, not the independent GNN
> baseline benchmark. For independent randomly initialized raw-graph baselines,
> use `run_site_holdout_end_to_end_benchmark.py`.

`run_site_holdout_legacy_representation_baselines.py` reproduces the historical
downstream baseline protocol on the existing site-held-out folds.

For every fold, the default workflow starts from raw graphs and trains:

1. the contrastive GCN/GINE encoders;
2. the generator and reconstructed representations;
3. the full-model downstream heads;
4. matched no-reconstruction heads using the newly created `embeddings.pt`.

Nothing except the audited site split is borrowed from an earlier run. The
experiment remains deliberately different from the supervised end-to-end
raw-graph benchmark:

- baseline heads consume fold-specific `embeddings.pt`;
- those embeddings were produced by the contrastively trained GCN/GINE
  encoders;
- unavailable embeddings are zero-imputed and accompanied by the historical
  availability indicators;
- the full-model heads consume `recon_demo.pt`;
- baseline and full-model heads use the same MLP/GRU implementations, optimizer
  settings, epochs, seed, and split;
- SPECT/DTI classification is binary Control versus PD;
- multimodal no-reconstruction progression and MRI progression are not
  evaluated, matching the historical task scope.

Therefore, the GCN/GINE rows in this experiment are **no-reconstruction
ablations of the contrastive representations**. They must not be described as
independent no-contrastive baselines.

## Background run

```bash
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  .venv-uv/bin/python \
  model/run_site_holdout_legacy_representation_baselines.py \
  --site_cv_root \
  model/logs/site_5fold_cv/site_5fold_seed42_20260724_170258 \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --output_dir model/results/legacy_site_holdout_end_to_end_seed42 \
  --device cuda \
  > legacy_site_holdout_end_to_end.log 2>&1 < /dev/null &

echo $!
```

Monitor:

```bash
tail -f legacy_site_holdout_end_to_end.log
```

The run is resumable at fold and checkpoint level. A fold is reused only after
its new output directory contains `embeddings.pt`, encoder weights, generator
weights, `recon_demo.pt`, and all five untouched-test downstream checkpoints.
Otherwise that fold's full pipeline is rerun. Baseline checkpoints are resumed
only when their sidecar contains the SHA-256 digest of the exact
`embeddings.pt` used to train them, preventing accidental reuse after the
representations change.

For diagnostic use only, `--reuse_existing_representations` restores the older
fast behavior. Do not pass it for the clean end-to-end experiment.

## Outputs

- `per_fold_metrics.csv`: untouched site-test metrics;
- `legacy_site_holdout_summary.csv`: mean, sample SD, and fold values;
- `legacy_site_holdout_table.tex`: manuscript-ready table;
- `run_manifest.json`: split, representation, and protocol audits;
- `fold_N/*/checkpoints`: resumable downstream checkpoints.
