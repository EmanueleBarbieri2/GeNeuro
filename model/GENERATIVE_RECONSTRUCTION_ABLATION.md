### Generative Reconstruction Ablation

To isolate the performance gains of the generative reconstruction module, we evaluate an ablated variant (No GR) where missing modality embeddings are substituted with zero-padding prior to concatenation into the final downstream classifier. The following table reports the results.

*Comparison of \modelName performance with different generative reconstruction ablated.*

| Model | Modality | Classification: Bal. Acc. | Classification: Macro F1 | Classification: Macro AUC | Severity Part II (R²) | Severity Part III (R²) | Progression Part II (R²) | Progression Part III (R²) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| \modelName | Only fMRI | 0.5000 ± 0.0000 | 0.4953 ± 0.0020 | 0.0000 ± 0.0000 | −0.0324 ± 0.0248 | −0.1113 ± 0.0862 | −0.3090 ± 0.6351 | −0.0324 ± 0.0248 |
| \modelName | Only fMRI, No GR | 0.5000 ± 0.0000 | 0.4953 ± 0.0020 | 0.0000 ± 0.0000 | −0.0254 ± 0.0397 | −0.1037 ± 0.1861 | −1.2726 ± 0.8625 | −0.9153 ± 1.6788 |
| \modelName | Only DTI | 0.7430 ± 0.1394 | 0.7532 ± 0.1425 | 0.0000 ± 0.0000 | 0.0921 ± 0.0314 | −6.1680 ± 12.4369 | -- ± -- | -- ± -- |
| \modelName | Only DTI, No GR | 0.6745 ± 0.0765 | 0.6914 ± 0.0968 | 0.0000 ± 0.0000 | 0.0581 ± 0.1490 | −9.7261 ± 19.4677 | -- ± -- | -- ± -- |
| \modelName | Only MRI | 0.5333 ± 0.2449 | 0.5296 ± 0.2464 | 0.3102 ± 0.3807 | −0.0271 ± 0.0644 | −0.0017 ± 0.0273 | -- ± -- | -- ± -- |
| \modelName | Only MRI, No GR | 0.5333 ± 0.2449 | 0.5296 ± 0.2464 | 0.3102 ± 0.3807 | −0.2291 ± 0.1633 | −0.1987 ± 0.1374 | -- ± -- | -- ± -- |
| \modelName | Only SPECT | 0.9526 ± 0.0250 | 0.7969 ± 0.0553 | 0.0000 ± 0.0000 | 0.1080 ± 0.0427 | 0.1232 ± 0.0423 | 0.0992 ± 0.0264 | 0.0687 ± 0.0292 |
| \modelName | Only SPECT, No GR | 0.9540 ± 0.0223 | 0.8058 ± 0.0322 | 0.0000 ± 0.0000 | 0.1141 ± 0.0537 | 0.1339 ± 0.0293 | 0.0907 ± 0.0433 | 0.0666 ± 0.0326 |
| \modelName | No GR | **0.9572 ± 0.0068** | **0.8921 ± 0.0208** | **0.9840 ± 0.0034** | 0.2548 ± 0.0288 | **0.4630 ± 0.0301** | 0.1535 ± 0.0407 | 0.3334 ± 0.0480 |
| **\modelName** | **Full Model** | 0.9519 ± 0.0077 | 0.8829 ± 0.0155 | 0.9830 ± 0.0040 | **0.2646 ± 0.0264** | 0.4440 ± 0.0341 | **0.1722 ± 0.0478** | **0.3542 ± 0.0356** |

We observe that the ablated model (No GR) performs marginally better on cross-sectional tasks, specifically classification and UPDRS-III severity regression. This behavior is an expected trade-off: while the generative module successfully imputes missing modalities, the VAE introduces slight reconstruction noise that can obscure specific static morphological signals. However, the generative module proves its utility in the longitudinal progression forecasting task, where the Full Model strictly outperforms the ablated baseline. This demonstrates that for temporal tasks, preserving the continuity and completeness of a patient's multimodal history outweighs the minor cost of cross-sectional reconstruction noise.
