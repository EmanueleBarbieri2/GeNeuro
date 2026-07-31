# Site-held-out missingness controls

`run_site_holdout_missingness_controls.py` runs only two additional controls:

1. a three-class multinomial logistic regression receiving the four binary
   SPECT/MRI/fMRI/DTI acquisition indicators and no imaging content;
2. a complete-case GCN/GINE model restricted to visits with all four modalities
   genuinely observed.

The complete-case model uses no zero filling, masks, contrastive embeddings, or
generative reconstruction. Every fold and downstream task initializes a new
set of four encoders and a new prediction head. Because there are no complete-
case Prodromal visits, its classification task is binary Control versus PD.

```bash
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  .venv-uv/bin/python model/run_site_holdout_missingness_controls.py \
  --site_cv_root model/logs/site_5fold_cv/site_5fold_seed42_20260724_170258 \
  --data_root data \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --output_dir model/results/site_holdout_missingness_controls_seed42 \
  --device cuda \
  > site_holdout_missingness_controls.log 2>&1 < /dev/null &

echo $!
tail -f site_holdout_missingness_controls.log
```

Outputs include `per_fold_metrics.csv`, `complete_case_counts.csv`,
`missingness_controls_summary.csv`, and `missingness_controls_table.tex`.
