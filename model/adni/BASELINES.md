# ADNI comparison policy

All numbers used for model comparison must come from the same subject set,
diagnosis task, outer folds, and leakage policy. Published headline numbers are
context, not entries in our results table, unless the method is rerun on these
matrices under this repository's splits.

## Implemented leakage-free controls

`run_classical_baselines.py` evaluates the upper-triangle connectome features
for fMRI, DTI, and early fusion. It includes a class-prior dummy classifier,
regularized logistic regression, PCA plus logistic regression, calibrated RBF
SVM, and random forest. Hyperparameters are selected by an inner stratified CV.
Scaling, PCA, calibration, and fitting occur entirely inside the corresponding
outer training partition.

`run_classification.py` provides three neural protocols:

- `frozen_contrastive`: the original label-free paired-modality pretraining,
  followed by a frozen-embedding classifier.
- `supervised`: both GINE encoders and the classifier are trained end to end.
- `joint`: contrastive pretraining followed by end-to-end classification with
  an auxiliary paired-modality contrastive loss.

Every neural protocol uses an inner validation partition for checkpoint
selection. The outer test fold is evaluated once.

`run_brainnetcnn.py` implements the published matrix-native E2E/E2N/N2G
design for fMRI, DTI, or a declared two-channel early-fusion adaptation. Every
edge and modality is standardized from the inner-training subjects only; the
saved fold checkpoint contains those training-fold normalization statistics.

## Published methods and comparability

| Method | Native evidence | Suitable comparison here |
| --- | --- | --- |
| BrainNetCNN | Matrix-native connectome CNN; established unimodal baseline | Implemented for fMRI and DTI, plus a declared two-channel adaptation |
| BrainGB | Modular brain-GNN benchmark rather than one fixed model | Rerun selected documented configurations on identical outer folds |
| BrainGNN | ROI-aware pooling model evaluated primarily on fMRI ASD/HCP tasks | fMRI-only architecture baseline after adapting its output head; published numbers are not directly comparable |
| MochaGCN | Paired AAL90 fMRI/DTI, but native AD-vs-CN experiment used 115 subjects (55 AD, 60 NC) | Architecturally relevant to `ad_vs_cn`; its reported 93.56% is not comparable to this 237-subject subset |
| BrainMAP | Released 407-subject matrices and three-class labels | Same cohort, but published/released preprocessing uses labels before outer CV; compare only after a fold-local reimplementation |

Primary sources:

- BrainNetCNN: <https://brainnetcnn.cs.sfu.ca/>
- BrainGB: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10079627/>
- BrainGNN: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9916535/>
- MochaGCN: <https://www.medrxiv.org/content/10.1101/2024.10.29.24316334v2.full>
- BrainMAP: <https://arxiv.org/abs/2506.11178>

## Required reporting

The primary metrics are balanced accuracy and macro-F1. Multiclass experiments
also report macro one-vs-rest ROC-AUC. Binary experiments additionally report
ROC-AUC, PR-AUC, disease sensitivity, and control specificity. Raw accuracy is
secondary because the AD-vs-CN majority baseline is 190/237 = 80.2%.

Binary tasks must be retrained independently; they are not calculated by
discarding one class from predictions made by a three-class model.
