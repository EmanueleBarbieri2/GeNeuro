"""Loading, validation, and graph construction for paired ADNI connectomes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .atlas import AAL90_REGIONS


LABEL_NAMES = {
    0: "Alzheimer's disease",
    1: "Mild cognitive impairment",
    2: "Normal control",
}
EXPECTED_COUNTS = {0: 47, 1: 170, 2: 190}


def _load_payload(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if not isinstance(payload, np.ndarray) or payload.shape != ():
        raise ValueError(f"{path} must contain a scalar object dictionary.")
    payload = payload.item()
    if not isinstance(payload, dict) or not {"data", "labels"}.issubset(payload):
        raise ValueError(f"{path} must contain 'data' and 'labels'.")
    matrices = np.asarray(payload["data"])
    labels = np.asarray(payload["labels"]).reshape(-1)
    return matrices, labels


def load_paired_connectomes(
    fmri_path: Union[str, Path],
    dti_path: Union[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmri, fmri_labels = _load_payload(fmri_path)
    dti, dti_labels = _load_payload(dti_path)

    expected_shape = (len(fmri_labels), len(AAL90_REGIONS), len(AAL90_REGIONS))
    if fmri.shape != expected_shape:
        raise ValueError(f"Unexpected fMRI shape {fmri.shape}; expected {expected_shape}.")
    if dti.shape != expected_shape:
        raise ValueError(f"Unexpected DTI shape {dti.shape}; expected {expected_shape}.")
    if not np.array_equal(fmri_labels, dti_labels):
        raise ValueError("fMRI and DTI label vectors differ; row-wise pairing is unsafe.")
    if not np.isfinite(fmri).all() or not np.isfinite(dti).all():
        raise ValueError("Connectomes contain NaN or infinite values.")
    if not np.allclose(fmri, fmri.transpose(0, 2, 1), atol=1e-6):
        raise ValueError("fMRI connectomes are not symmetric.")
    if not np.allclose(dti, dti.transpose(0, 2, 1), atol=1e-6):
        raise ValueError("DTI connectomes are not symmetric.")

    labels = fmri_labels.astype(np.int64, copy=False)
    counts = dict(Counter(labels.tolist()))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected ADNI label counts {counts}; expected {EXPECTED_COUNTS}.")
    return fmri.astype(np.float32), dti.astype(np.float32), labels


def matrix_to_graph(matrix: np.ndarray, modality: str) -> Data:
    """Represent each ROI by its connectivity profile and nonzero edges."""
    weights = torch.from_numpy(np.asarray(matrix, dtype=np.float32).copy())
    if modality == "DTI":
        # Preserve the published sparsity and compress the positive dynamic range.
        weights = torch.log1p(weights.clamp_min(0))
        edge_mask = weights > 0
    elif modality == "fMRI":
        # Values are signed and appear Fisher-z transformed; do not clip them.
        edge_mask = weights != 0
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    edge_mask.fill_diagonal_(False)
    edge_index = edge_mask.nonzero(as_tuple=False).t().contiguous()
    edge_attr = weights[edge_mask].contiguous()
    return Data(
        x=weights,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=weights.size(0),
    )


class PairedADNIDataset(Dataset):
    def __init__(
        self,
        fmri: np.ndarray,
        dti: np.ndarray,
        labels: np.ndarray,
        indices: Optional[Iterable[int]] = None,
        augment: bool = False,
        edge_drop_prob: float = 0.15,
        feature_jitter: float = 0.01,
    ):
        self.fmri = fmri
        self.dti = dti
        self.labels = labels
        self.indices = np.asarray(
            list(indices) if indices is not None else np.arange(len(labels)), dtype=np.int64
        )
        self.augment = augment
        self.edge_drop_prob = edge_drop_prob
        self.feature_jitter = feature_jitter
        # Graph topology is deterministic for a subject and expensive to rebuild
        # every epoch, so cache it once per dataset partition.
        self._graphs = {
            int(index): (
                matrix_to_graph(self.fmri[int(index)], "fMRI"),
                matrix_to_graph(self.dti[int(index)], "DTI"),
            )
            for index in self.indices
        }

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, graph: Data) -> Data:
        if not self.augment:
            return graph
        graph = graph.clone()
        if self.edge_drop_prob > 0 and graph.edge_index.size(1):
            keep = torch.rand(graph.edge_index.size(1)) >= self.edge_drop_prob
            # Avoid pathological empty graphs.
            if keep.any():
                graph.edge_index = graph.edge_index[:, keep]
                graph.edge_attr = graph.edge_attr[keep]
        if self.feature_jitter > 0:
            scale = graph.x.detach().std().clamp_min(1e-6)
            graph.x = graph.x + torch.randn_like(graph.x) * scale * self.feature_jitter
        return graph

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        base_fmri, base_dti = self._graphs[index]
        fmri = self._augment(base_fmri)
        dti = self._augment(base_dti)
        return {
            "index": index,
            "fmri": fmri,
            "dti": dti,
            "label": int(self.labels[index]),
        }


def paired_collate(samples):
    return {
        "index": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
        "fmri": Batch.from_data_list([sample["fmri"] for sample in samples]),
        "dti": Batch.from_data_list([sample["dti"] for sample in samples]),
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
    }
