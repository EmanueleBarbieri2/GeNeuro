# Missingness-only downstream audit

`run_missingness_audit.py` measures how much downstream performance can be
obtained from acquisition patterns without using any imaging embedding values.
It reads the four indicators derived from `recon_demo.pt` in this order:

```text
SPECT, MRI, fMRI, DTI
```

An indicator is zero when the modality was genuinely observed and one when it
was missing and reconstructed.

The audit trains and evaluates:

- a class-balanced multinomial logistic regression for diagnosis;
- four additional diagnostic classifiers using one modality-availability
  indicator at a time, directly identifying which acquisition indicator is
  predictive;
- ridge regressors for static UPDRS-II and UPDRS-III;
- lightweight GRUs for longitudinal UPDRS-II and UPDRS-III using:
  - missingness sequences only;
  - visit timing only;
  - missingness sequences plus visit timing.

Class-prior and training-mean controls are also reported. The existing
train/validation/test split is reused, and execution stops if any patient
appears in more than one partition. If a test partition exists it is used for
final evaluation; otherwise validation performance is reported explicitly.

## Run

```bash
CUDA_VISIBLE_DEVICES=1 python3 model/run_missingness_audit.py \
  --checkpoints_dir model/checkpoints/tsne_fold0 \
  --data_csv data/PPMI_Curated_Data_Cut_Public_20251112.csv \
  --split_path data/unified_split_master.txt \
  --output_dir model/results/missingness_audit \
  --device cuda
```

The models are small, so CPU execution is also practical:

```bash
python3 model/run_missingness_audit.py \
  --checkpoints_dir model/checkpoints/tsne_fold0 \
  --split_path data/unified_split_master.txt \
  --device cpu
```

## Outputs

- `metrics.csv`: reviewer-facing metrics and patient-bootstrap intervals;
- `predictions.csv`: sample-level predictions for independent checking;
- `coefficients.csv`: logistic/ridge coefficients for each modality indicator;
- `availability_by_class.csv`: observed modality rates by class and partition;
- `missingness_patterns_by_class.csv`: frequencies of all four-bit patterns;
- `summary.json`: complete configuration, split counts, and metrics.

The progression ensemble row averages predictions across the requested seeds.
Individual seed rows are retained to expose optimization variability.

This experiment quantifies predictive information in acquisition patterns. It
does not by itself prove that embedding-only performance is pathological:
generated embeddings may still reveal whether a modality was reconstructed,
and classes without overlap in modality availability cannot be fully
disentangled using this dataset alone.
