### Site-held-out downstream results

Values are mean ± sample standard deviation across five outer folds. Each test
fold contains sites absent from both training and validation. Unimodal
baselines use only genuinely observed scans. The complete-case GCN/GINE
baseline includes only visits with all four modalities genuinely observed and
uses no imputation, zero filling, or availability indicators. The full model
uses contrastive alignment and generative reconstruction of unavailable
modalities. Classification results marked with † are binary Control-versus-PD;
all other classification results are three-class. Binary and three-class
results are not directly comparable. A dash indicates that an endpoint could
not be evaluated validly across all five folds.

| Model | Input / modality | Classification: balanced accuracy | Classification: macro F1 | Classification: macro AUC | Severity Part II (R²) | Severity Part III (R²) | Progression Part II (R²) | Progression Part III (R²) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | SPECT† | 0.9406 ± 0.0254 | 0.8812 ± 0.0362 | 0.9778 ± 0.0244 | 0.1636 ± 0.0291 | 0.2378 ± 0.0215 | 0.0780 ± 0.0463 | −0.0001 ± 0.0330 |
| GCN | MRI | 0.3372 ± 0.0226 | 0.2684 ± 0.0596 | 0.5218 ± 0.0203 | −0.0093 ± 0.0222 | −0.0239 ± 0.0320 | — | — |
| GINE | fMRI | 0.3516 ± 0.0368 | 0.1614 ± 0.0532 | 0.5281 ± 0.0311 | −0.0213 ± 0.0189 | −0.0369 ± 0.0425 | −0.0879 ± 0.0532 | −0.0540 ± 0.1153 |
| GINE | DTI† | 0.5629 ± 0.0130 | 0.3950 ± 0.0730 | 0.6097 ± 0.0294 | −0.0169 ± 0.0136 | −0.0416 ± 0.0646 | 0.0095 ± 0.0368 | −0.0309 ± 0.0537 |
| GCN & GINE | All four observed† | 0.8128 ± 0.0729 | 0.7751 ± 0.0628 | 0.9481 ± 0.0332 | −0.2373 ± 0.4045 | −0.3689 ± 0.4277 | — | — |
| **GeNeuro** | **Multimodal, reconstructed** | **0.9333 ± 0.0168** | **0.8619 ± 0.0177** | **0.9786 ± 0.0086** | **0.2290 ± 0.0499** | **0.4129 ± 0.0687** | **0.1570 ± 0.0216** | **0.2797 ± 0.0626** |

† Binary Control-versus-PD classification. No Prodromal visit has the required
SPECT, DTI, or complete all-four-modality acquisition pattern.
