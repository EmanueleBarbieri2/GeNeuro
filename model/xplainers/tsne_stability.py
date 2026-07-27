#!/usr/bin/env python3
"""Generate reproducible PCA-initialized t-SNE figures and multi-seed audits."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


DEFAULT_SEEDS = list(range(10))
MARKERS = ("o", "^", "s", "D", "P", "X", "v", "<", ">")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base_dir, path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))


def _load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("panels"), list) or not manifest["panels"]:
        raise ValueError("Manifest must contain a non-empty 'panels' list.")
    return manifest


def _load_panel(panel, manifest_dir):
    input_path = _resolve(manifest_dir, panel["input"])
    if not input_path.endswith(".npz"):
        raise ValueError(f"{panel['name']}: only safe, portable .npz inputs are supported.")
    with np.load(input_path, allow_pickle=False) as payload:
        feature_key = panel.get("feature_key", "features")
        if feature_key not in payload:
            raise KeyError(f"{panel['name']}: feature key {feature_key!r} not found in {input_path}")
        features = np.asarray(payload[feature_key], dtype=np.float64)
        color_key = panel.get("color_key")
        style_key = panel.get("style_key")
        id_key = panel.get("id_key", "ids")
        colors = np.asarray(payload[color_key]) if color_key else None
        styles = np.asarray(payload[style_key]) if style_key else None
        ids = np.asarray(payload[id_key]).astype(str) if id_key in payload else np.arange(len(features)).astype(str)

    if features.ndim != 2:
        raise ValueError(f"{panel['name']}: features must have shape [samples, dimensions].")
    if len(features) < 3:
        raise ValueError(f"{panel['name']}: at least three samples are required.")
    if not np.isfinite(features).all():
        raise ValueError(f"{panel['name']}: features contain NaN or infinite values.")
    for key, values in ((panel.get("color_key"), colors), (panel.get("style_key"), styles), ("ids", ids)):
        if values is not None and len(values) != len(features):
            raise ValueError(f"{panel['name']}: {key} length does not match the feature matrix.")

    return {
        "input_path": input_path,
        "features": features,
        "colors": colors,
        "styles": styles,
        "ids": ids,
    }


def _standardize(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.where(std > 0, std, 1.0)


def _fit_tsne(features, config, init, seed):
    perplexity = float(config.get("perplexity", 30.0))
    if perplexity >= len(features):
        raise ValueError(f"perplexity ({perplexity}) must be smaller than n_samples ({len(features)}).")
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=config.get("learning_rate", "auto"),
        max_iter=int(config.get("max_iter", 1000)),
        early_exaggeration=float(config.get("early_exaggeration", 12.0)),
        init=init,
        random_state=int(seed),
        metric=config.get("metric", "euclidean"),
        method=config.get("method", "barnes_hut"),
        angle=float(config.get("angle", 0.5)),
    )
    return model.fit_transform(features), float(model.kl_divergence_)


def _neighbors(coordinates, n_neighbors):
    count = min(int(n_neighbors), len(coordinates) - 1)
    indices = NearestNeighbors(n_neighbors=count + 1).fit(coordinates).kneighbors(return_distance=False)
    return [set(row[row != index][:count]) for index, row in enumerate(indices)]


def _neighbor_overlap(left, right):
    scores = []
    for left_set, right_set in zip(left, right):
        union = left_set | right_set
        scores.append(len(left_set & right_set) / len(union) if union else 1.0)
    return float(np.mean(scores))


def _categorical_silhouette(coordinates, labels):
    if labels is None:
        return math.nan
    labels = labels.astype(str)
    counts = np.unique(labels, return_counts=True)[1]
    if len(counts) < 2 or np.any(counts < 2) or len(counts) >= len(labels):
        return math.nan
    return float(silhouette_score(coordinates, labels))


def _mean_std(values):
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite))}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _plot(ax, coordinates, panel, colors, styles, category_colors=None):
    color_mode = panel.get("color_mode", "categorical")
    title = panel.get("title", panel["name"])
    point_size = float(panel.get("point_size", 10))
    alpha = float(panel.get("alpha", 0.72))

    if colors is None:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=point_size, alpha=alpha, linewidths=0)
    elif color_mode == "continuous":
        numeric_colors = np.asarray(colors, dtype=float)
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=numeric_colors,
            cmap=panel.get("cmap", "viridis"),
            vmin=float(panel.get("vmin", np.nanmin(numeric_colors))),
            vmax=float(panel.get("vmax", np.nanmax(numeric_colors))),
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
        plt.colorbar(scatter, ax=ax, label=panel.get("color_label", panel.get("color_key", "Value")))
    else:
        string_colors = colors.astype(str)
        categories = sorted(np.unique(string_colors))
        if category_colors is None:
            palette = plt.get_cmap(panel.get("cmap", "tab10"))
            category_colors = {category: palette(index % palette.N) for index, category in enumerate(categories)}
        style_values = styles.astype(str) if styles is not None else np.repeat("all", len(coordinates))
        style_categories = sorted(np.unique(style_values))
        for style_index, style in enumerate(style_categories):
            for category in categories:
                selected = (style_values == style) & (string_colors == category)
                if not selected.any():
                    continue
                ax.scatter(
                    coordinates[selected, 0],
                    coordinates[selected, 1],
                    color=category_colors[category],
                    marker=MARKERS[style_index % len(MARKERS)],
                    s=point_size,
                    alpha=alpha,
                    linewidths=0,
                    label=category if styles is None else f"{category} | {style}",
                )
        ax.legend(loc="best", fontsize=7, frameon=False, markerscale=1.5)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _write_coordinates(path, ids, coordinates, colors, styles, init, seed):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "tsne_1", "tsne_2", "color", "style", "initialization", "seed"])
        for index, sample_id in enumerate(ids):
            writer.writerow(
                [
                    sample_id,
                    float(coordinates[index, 0]),
                    float(coordinates[index, 1]),
                    "" if colors is None else colors[index],
                    "" if styles is None else styles[index],
                    init,
                    seed,
                ]
            )


def _process_panel(panel, data, global_config, seeds, output_dir):
    name = panel["name"]
    panel_dir = os.path.join(output_dir, name)
    coordinates_dir = os.path.join(panel_dir, "coordinates")
    os.makedirs(coordinates_dir, exist_ok=True)

    config = dict(global_config)
    config.update(panel.get("tsne", {}))
    features = _standardize(data["features"]) if panel.get("standardize", False) else data["features"]
    # sklearn's trustworthiness requires n_neighbors < n_samples / 2.
    n_neighbors = min(int(config.get("n_neighbors", 10)), max(1, (len(features) - 1) // 2))
    color_mode = panel.get("color_mode", "categorical")
    labels = data["colors"] if color_mode == "categorical" else None

    runs = []
    main_seed = int(config.get("main_seed", 42))
    main_coordinates, main_kl = _fit_tsne(features, config, "pca", main_seed)
    runs.append(
        {
            "name": "pca",
            "initialization": "pca",
            "seed": main_seed,
            "coordinates": main_coordinates,
            "kl_divergence": main_kl,
        }
    )
    for seed in seeds:
        coordinates, kl_divergence = _fit_tsne(features, config, "random", seed)
        runs.append(
            {
                "name": f"random_seed_{seed}",
                "initialization": "random",
                "seed": seed,
                "coordinates": coordinates,
                "kl_divergence": kl_divergence,
            }
        )

    category_colors = None
    if labels is not None:
        palette = plt.get_cmap(panel.get("cmap", "tab10"))
        category_colors = {
            category: palette(index % palette.N)
            for index, category in enumerate(sorted(np.unique(labels.astype(str))))
        }

    main_figure, main_axis = plt.subplots(figsize=tuple(panel.get("figsize", [7.0, 5.5])))
    _plot(main_axis, main_coordinates, panel, data["colors"], data["styles"], category_colors)
    main_axis.set_title(f"{panel.get('title', name)} (PCA initialization, seed {main_seed})")
    main_figure.tight_layout()
    main_path = os.path.join(panel_dir, f"{name}.png")
    main_figure.savefig(main_path, dpi=int(panel.get("dpi", 300)), bbox_inches="tight")
    plt.close(main_figure)

    columns = int(math.ceil(math.sqrt(len(seeds))))
    rows = int(math.ceil(len(seeds) / columns))
    seed_figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False)
    for axis, run in zip(axes.flat, runs[1:]):
        _plot(axis, run["coordinates"], panel, data["colors"], data["styles"], category_colors)
        axis.set_title(f"Random initialization, seed {run['seed']}")
    for axis in axes.flat[len(seeds):]:
        axis.axis("off")
    seed_figure.suptitle(f"{panel.get('title', name)}: initialization sensitivity", fontsize=14)
    seed_figure.tight_layout()
    seed_grid_path = os.path.join(panel_dir, f"{name}_random_initializations.png")
    seed_figure.savefig(seed_grid_path, dpi=int(panel.get("dpi", 300)), bbox_inches="tight")
    plt.close(seed_figure)

    neighborhoods = [_neighbors(run["coordinates"], n_neighbors) for run in runs]
    metric_rows = []
    for index, run in enumerate(runs):
        overlaps = [
            _neighbor_overlap(neighborhoods[index], neighborhoods[other])
            for other in range(len(runs))
            if other != index
        ]
        metric_rows.append(
            {
                "initialization": run["initialization"],
                "seed": run["seed"],
                "kl_divergence": run["kl_divergence"],
                "trustworthiness": float(
                    trustworthiness(features, run["coordinates"], n_neighbors=n_neighbors)
                ),
                "silhouette": _categorical_silhouette(run["coordinates"], labels),
                "mean_neighbor_jaccard": float(np.mean(overlaps)),
                "neighbor_jaccard_vs_pca": (
                    1.0 if index == 0 else _neighbor_overlap(neighborhoods[index], neighborhoods[0])
                ),
            }
        )
        coordinate_path = os.path.join(
            coordinates_dir,
            f"{name}_{run['initialization']}_seed{run['seed']}.csv",
        )
        _write_coordinates(
            coordinate_path,
            data["ids"],
            run["coordinates"],
            data["colors"],
            data["styles"],
            run["initialization"],
            run["seed"],
        )

    metrics_path = os.path.join(panel_dir, f"{name}_stability_metrics.csv")
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    random_rows = [row for row in metric_rows if row["initialization"] == "random"]
    summary = {
        "name": name,
        "input": data["input_path"],
        "input_sha256": _sha256(data["input_path"]),
        "samples": int(features.shape[0]),
        "dimensions": int(features.shape[1]),
        "main_figure": main_path,
        "random_initialization_figure": seed_grid_path,
        "metrics_csv": metrics_path,
        "pca_initialization": metric_rows[0],
        "random_initializations": {
            metric: _mean_std([row[metric] for row in random_rows])
            for metric in ("kl_divergence", "trustworthiness", "silhouette", "mean_neighbor_jaccard")
        },
        "tsne": config,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Create PCA-initialized t-SNE figures and multi-seed robustness analyses."
    )
    parser.add_argument("--manifest", required=True, help="JSON manifest describing all latent-space panels.")
    parser.add_argument("--output_dir", help="Override the output directory in the manifest.")
    parser.add_argument("--seeds", help="Comma-separated random initialization seeds (default: 0-9).")
    parser.add_argument("--panels", help="Comma-separated panel names to process.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    manifest_dir = os.path.dirname(manifest_path)
    manifest = _load_manifest(manifest_path)
    output_dir = os.path.abspath(
        args.output_dir
        or _resolve(manifest_dir, manifest.get("output_dir", "tsne_stability_output"))
    )
    if os.path.isdir(output_dir) and os.listdir(output_dir) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite or choose another directory."
        )
    os.makedirs(output_dir, exist_ok=True)

    seeds = (
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [int(value) for value in manifest.get("random_seeds", DEFAULT_SEEDS)]
    )
    if not seeds:
        raise ValueError("At least one random initialization seed is required.")
    selected_panels = set(args.panels.split(",")) if args.panels else None

    summaries = []
    for panel in manifest["panels"]:
        if selected_panels is not None and panel["name"] not in selected_panels:
            continue
        print(f"Processing {panel['name']}...")
        data = _load_panel(panel, manifest_dir)
        summaries.append(
            _process_panel(
                panel,
                data,
                manifest.get("tsne", {}),
                seeds,
                output_dir,
            )
        )

    if not summaries:
        raise ValueError("No panels were selected.")
    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "output_dir": output_dir,
        "random_seeds": seeds,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "panels": summaries,
    }
    summary_path = os.path.join(output_dir, "tsne_stability_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(_json_safe(run_summary), handle, indent=2, allow_nan=False)
    print(f"Saved reproducibility summary: {summary_path}")


if __name__ == "__main__":
    main()
