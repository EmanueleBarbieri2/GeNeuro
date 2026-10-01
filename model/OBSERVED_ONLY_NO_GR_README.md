# No-GR observed-only ablation

This is the default No-GR condition for the site-held-out ablation suite. At
each visit it mean-pools only the contrastively aligned embeddings of modalities
that were genuinely acquired. It does not reconstruct, zero-fill, concatenate
availability indicators, or require a complete four-modality visit.

Mean pooling is used because all encoders produce 1,024-dimensional embeddings
in the shared contrastive space. It keeps the representation width fixed while
avoiding magnitude-based encoding of the number of available modalities.

Run the complete matched No-CL/No-GR suite:

```bash
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  .venv-uv/bin/python model/run_site_holdout_ablation_suite.py \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --output_root model/logs/site_holdout_ablations_observed_only \
  --device cuda \
  --no_gr_strategy observed_only \
  > model/logs/site_holdout_ablations_observed_only.log 2>&1 < /dev/null &

echo $! | tee model/logs/site_holdout_ablations_observed_only.pid
```

Monitor it with:

```bash
tail -f model/logs/site_holdout_ablations_observed_only.log
```

To run only the five-fold observed-only No-GR condition:

```bash
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  .venv-uv/bin/python model/run_5fold_site_cv.py \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --logs_dir model/logs/no_gr_observed_only \
  --device cuda \
  --disable_generator \
  --observed_only_pooling \
  --no_missingness_mask \
  > model/logs/no_gr_observed_only.log 2>&1 < /dev/null &

echo $! | tee model/logs/no_gr_observed_only.pid
```

The runner rejects `--observed_only_pooling` unless both
`--disable_generator` and `--no_missingness_mask` are present.
