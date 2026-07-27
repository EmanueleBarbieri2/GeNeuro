# Reproducible t-SNE stability analysis

`tsne_stability.py` regenerates every latent-space panel with:

- a PCA-initialized main t-SNE using a fixed seed;
- a grid of independent random initializations;
- projected classification regions, severity surfaces, and longitudinal
  trajectories on both the main panel and every random-initialization panel;
- trustworthiness, silhouette, KL-divergence, and neighborhood-overlap statistics;
- CSV files containing the two-dimensional coordinates for every run;
- a compact `tsne_stability_rebuttal_table.csv` with mean and standard
  deviation values ready to report;
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
`classifier.pt`, both static severity checkpoints, and both progression
checkpoints. It writes:

- encoder outputs to `realEmb.npz`;
- observed and reconstructed modality embeddings to `genEmb.npz`;
- classifier penultimate-layer representations to `classEmb.npz`;
- static-regression penultimate-layer representations to
  `sevEmb_U2_ADL.npz` and `sevEmb_U3_Motor.npz`;
- final GRU hidden states to `progEmb_U2_ADL.npz` and
  `progEmb_U3_Motor.npz`.

The classification background is fitted to the classifier's predicted class,
the severity background to the regressor's predicted score, and the progression
background to the GRU's predicted future score. Patient-level progression
samples are connected in chronological rolling-window order.

By default, both Part II and Part III are exported. Use
`--severity_target U2_ADL` or `--progression_target U2_ADL` to limit an export
to one target. The supplied example manifest points to all generated files
under `data/visualizations`.

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

The colored backgrounds are deliberately described as **projected prediction
regions/surfaces**, not exact decision boundaries. t-SNE has no inverse
transformation, so the script interpolates predictions already made by the
trained downstream model in the displayed 2-D coordinates using
distance-weighted nearest neighbors. This is a visualization aid and is not
used to compute downstream performance.
