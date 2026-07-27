# Reproducible t-SNE stability analysis

`tsne_stability.py` regenerates every latent-space panel with:

- a PCA-initialized main t-SNE using a fixed seed;
- a grid of independent random initializations;
- trustworthiness, silhouette, KL-divergence, and neighborhood-overlap statistics;
- CSV files containing the two-dimensional coordinates for every run;
- SHA-256 hashes of the manifest and source arrays.

## Input artifacts

The five input archives can be extracted directly from a completed pipeline run:

```bash
python3 model/xplainers/prepare_tsne_inputs.py \
  --checkpoints_dir model/checkpoints \
  --split_path data/unified_split_fold0.txt \
  --partition val
```

For a site-held-out run, use that run's checkpoint directory and split file with
`--partition test`. The exporter reads `embeddings.pt`, `recon_demo.pt`,
`classifier.pt`, `static_U3_Motor.pt`, and `prog_U3_Motor.pt`. It writes:

- encoder outputs to `realEmb.npz`;
- observed and reconstructed modality embeddings to `genEmb.npz`;
- classifier penultimate-layer representations to `classEmb.npz`;
- static-regression penultimate-layer representations to `sevEmb.npz`;
- final GRU hidden states to `progEmb.npz`.

Use `--severity_target U2_ADL` or `--progression_target U2_ADL` to visualize Part
II instead of the default Part III. The supplied example manifest already
points to the generated files under `data/visualizations`.

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
