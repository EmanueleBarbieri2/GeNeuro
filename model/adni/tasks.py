"""Diagnosis tasks for the paired ADNI connectome cohort."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOURCE_LABEL_NAMES = {
    0: "Alzheimer's disease",
    1: "Mild cognitive impairment",
    2: "Normal control",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source_to_target: dict[int, int]
    label_names: dict[int, str]
    description: str

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


TASKS = {
    "three_class": TaskSpec(
        name="three_class",
        source_to_target={0: 0, 1: 1, 2: 2},
        label_names=SOURCE_LABEL_NAMES,
        description="Alzheimer's disease vs mild cognitive impairment vs normal control",
    ),
    "ad_vs_cn": TaskSpec(
        name="ad_vs_cn",
        source_to_target={2: 0, 0: 1},
        label_names={0: "Normal control", 1: "Alzheimer's disease"},
        description="Alzheimer's disease vs normal control",
    ),
    "mci_vs_cn": TaskSpec(
        name="mci_vs_cn",
        source_to_target={2: 0, 1: 1},
        label_names={0: "Normal control", 1: "Mild cognitive impairment"},
        description="Mild cognitive impairment vs normal control",
    ),
    "ad_vs_mci": TaskSpec(
        name="ad_vs_mci",
        source_to_target={1: 0, 0: 1},
        label_names={0: "Mild cognitive impairment", 1: "Alzheimer's disease"},
        description="Alzheimer's disease vs mild cognitive impairment",
    ),
}


def prepare_task_labels(source_labels: np.ndarray, task_name: str) -> tuple[np.ndarray, np.ndarray, TaskSpec]:
    """Return full-length remapped labels and eligible original subject indices."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}; choose from {sorted(TASKS)}.")
    spec = TASKS[task_name]
    labels = np.full(len(source_labels), -1, dtype=np.int64)
    for source, target in spec.source_to_target.items():
        labels[source_labels == source] = target
    eligible = np.flatnonzero(labels >= 0)
    if not len(eligible):
        raise ValueError(f"Task {task_name!r} has no eligible subjects.")
    return labels, eligible, spec
