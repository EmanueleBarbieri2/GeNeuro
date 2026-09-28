# ADNI AAL90 classification

This is the classification-only path for the paired `ADNI_fMRI.npy` and
`ADNI_DTI.npy` files released with BrainMAP. It deliberately does not invoke
the PPMI MRI/SPECT, UPDRS, longitudinal, site-holdout, or generative branches.
The source files and their canonical ROI order are available in the
[BrainMAP repository](https://github.com/danleneurocom/BrainMAP), which
accompanies the IEEE paper
[BrainMAP: Multimodal Graph Learning for Efficient Brain Disease Localization](https://ieeexplore.ieee.org/document/11589209).

The source data use this diagnosis encoding:

- `0`: Alzheimer's disease (47 subjects)
- `1`: mild cognitive impairment (170 subjects)
- `2`: normal control (190 subjects)

Rows are paired across modalities and each of the 407 rows is a distinct
participant. The 90 matrix positions use the canonical AAL90 order recorded in
`atlas.py` and exported into every run directory as `aal90_regions.csv`.

## Evaluation design

The runner performs shuffled stratified outer cross-validation. Inside every
outer fold, the development subjects are split into training and validation
partitions. The two encoders see only the inner training subjects; the
validation partition selects the classifier checkpoint, and the outer test
partition is touched once for final fold metrics. This avoids the supervised
pre-CV subgraph-selection and feature-distillation leakage present in the
reference BrainMAP script.

DTI zero entries are omitted from message passing rather than represented as
edges. This is important because approximately 86.5% of entries in the
provided DTI matrices are zero.

## Run

From the repository root:

```bash
python -m model.adni.run_classification \
  --device cuda \
  --output_dir model/checkpoints/adni_cv
```

Use `--device cpu` on a machine without CUDA. Outputs include one checkpoint
per fold, a subject manifest, the AAL90 order, per-subject out-of-fold
predictions, and aggregate metrics in `summary.json`.
