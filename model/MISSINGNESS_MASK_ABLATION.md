# Explicit missingness-mask ablation

This ablation tests whether the downstream heads rely on the four binary values
that indicate whether SPECT, MRI, fMRI, and DTI were observed or reconstructed.
The encoder and generator outputs are held fixed between conditions.

## 1. Train the normal pipeline

```bash
python3 model/run_full_pipeline.py \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --split_path data/unified_split_fold0.txt \
  --checkpoints_dir model/checkpoints/mask_comparison_fold0/with_mask \
  --device cuda
```

This produces the shared `recon_demo.pt` and trains the normal downstream heads
with the four indicators.

## 2. Reuse the exact representations and remove the indicators

```bash
python3 model/run_full_pipeline.py \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --split_path data/unified_split_fold0.txt \
  --reuse_representations_dir model/checkpoints/mask_comparison_fold0/with_mask \
  --checkpoints_dir model/checkpoints/mask_comparison_fold0/without_mask \
  --no_missingness_mask \
  --device cuda
```

Only the downstream heads are retrained in the second command. Their input is
the same concatenated modality representation, reduced from 4100 dimensions
(four 1024-dimensional embeddings plus four indicators) to 4096 dimensions.

Compare the `metrics` dictionaries in `classifier.pt`, `static_U2_ADL.pt`,
`static_U3_Motor.pt`, `prog_U2_ADL.pt`, and `prog_U3_Motor.pt` across the two
directories. Each checkpoint records `use_missingness_mask` for auditing.

This isolates reliance on the explicit indicators. It does not guarantee that
the reconstructed embeddings themselves contain no detectable reconstruction
signature, so it should be reported alongside the missingness-only audit.
