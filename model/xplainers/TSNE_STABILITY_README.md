# Reproducible t-SNE stability analysis

`tsne_stability.py` regenerates every latent-space panel with:

- a PCA-initialized main t-SNE using a fixed seed;
- a grid of independent random initializations;
- trustworthiness, silhouette, KL-divergence, and neighborhood-overlap statistics;
- CSV files containing the two-dimensional coordinates for every run;
- SHA-256 hashes of the manifest and source arrays.

## Input artifacts

Save the exact arrays used by the original figures as uncompressed or compressed
NumPy archives. Each archive must contain a two-dimensional `features` array and
may contain string or numeric annotation arrays:

```python
np.savez_compressed(
    "realEmb.npz",
    features=embedding_matrix,
    modality=modality_labels,
    ids=subject_ids,
)
```

The supplied `tsne_stability_manifest.example.json` documents the expected keys
for `realEmb`, `genEmb`, `classEmb`, `sevEmb`, and `progEmb`. Copy it to a new
manifest and change the paths or keys to match the arrays used for the paper.
Paths are resolved relative to the manifest.

## Run

```bash
python3 model/xplainers/tsne_stability.py \
  --manifest model/xplainers/tsne_stability_manifest.json
```

To test one panel and fewer seeds:

```bash
python3 model/xplainers/tsne_stability.py \
  --manifest model/xplainers/tsne_stability_manifest.json \
  --panels classEmb \
  --seeds 0,1,2
```

The script refuses to write into a non-empty output directory unless
`--overwrite` is explicitly provided.

## Interpretation

The PCA-initialized image is the reproducible main-text or supplementary figure.
The random-seed grid addresses initialization sensitivity directly. Report the
mean and standard deviation of trustworthiness and silhouette across random
initializations. `mean_neighbor_jaccard` quantifies the consistency of local
two-dimensional neighborhoods across runs and is invariant to rotation and
reflection of the plot.
