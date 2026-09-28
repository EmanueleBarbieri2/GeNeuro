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
  --task three_class \
  --training_mode frozen_contrastive \
  --output_dir model/checkpoints/adni_cv
```

Available tasks are `three_class`, `ad_vs_cn`, `mci_vs_cn`, and `ad_vs_mci`.
Binary labels are always encoded as control/earlier stage `0` and disease/later
stage `1`, so sensitivity consistently refers to the disease-positive class.

For supervised end-to-end training:

```bash
python -m model.adni.run_classification \
  --device cuda \
  --task ad_vs_cn \
  --training_mode supervised \
  --output_dir model/checkpoints/adni_ad_vs_cn_supervised
```

For nested-CV connectome baselines across every task and modality:

```bash
python -m model.adni.run_classical_baselines \
  --task all \
  --n_jobs -1 \
  --output_dir model/checkpoints/adni_baselines
```

For a fold-normalized BrainNetCNN-style connectome baseline:

```bash
python -m model.adni.run_brainnetcnn \
  --device cuda \
  --task ad_vs_cn \
  --modality early_fusion \
  --output_dir model/checkpoints/adni_brainnetcnn_ad_vs_cn
```

## DTI-to-fMRI variational reconstruction

The ADNI-specific missing-modality experiment adapts the repository's
variational generator to the two available modalities. It first learns paired
fMRI/DTI graph embeddings on each inner-training partition. The VAE is then
trained and validated with fMRI explicitly masked, making its fMRI output
conditional on DTI alone. Diagnosis labels are used only by the downstream
classifiers.

```bash
python -u -m model.adni.run_vae_reconstruction \
  --device cuda \
  --task ad_vs_cn \
  --output_dir model/checkpoints/adni_vae_ad_vs_cn
```

The output compares capacity-matched classifiers using DTI, posterior-mean
generated fMRI, the posterior mean latent, real fMRI, and real fMRI+DTI. It
also reports a Monte Carlo prediction ensemble from 20 stochastic fMRI draws
and held-out reconstruction similarity. `summary.json` includes paired
outer-fold metric differences between generated fMRI and DTI, including the
number of folds in which reconstruction wins. The generated objects are fMRI
**embeddings**, matching the original generator; they are not 90x90 synthetic
connectivity matrices.

To run this experiment for every classification task through the resumable
suite:

```bash
python -u -m model.adni.run_benchmark_suite \
  --stages vae \
  --device cuda \
  --output_dir model/checkpoints/adni_benchmark
```

To run or resume the complete suite, including every task, training mode, and
modality baseline:

```bash
python -u -m model.adni.run_benchmark_suite \
  --device cuda \
  --output_dir model/checkpoints/adni_benchmark
```

The suite skips a job when its `summary.json` already exists, making it safe to
restart after a scheduler timeout or disconnected session. Use `--force` only
when completed jobs should be overwritten.

See [BASELINES.md](BASELINES.md) for the comparability audit and reporting
policy, and [RESULTS.md](RESULTS.md) for the initial nested-CV reference
results.

Use `--device cpu` on a machine without CUDA. Outputs include one checkpoint
per fold, a subject manifest, the AAL90 order, per-subject out-of-fold
predictions, and aggregate metrics in `summary.json`.
