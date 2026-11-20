"""JSON loader for the dataset-generation pipeline.

Reads a configuration file describing one or more datasets and turns it into a
list of validated `DatasetGenerateRequest` objects (the same model the web
backend consumes). The JSON format supports a `shared` block whose values act
as defaults for every entry in `datasets`; per-dataset fields override `shared`
for scalars and replace the entire list for `motifs`, `labeling_functions`,
and `perturbations`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from web.backend.models.dataset_models import DatasetGenerateRequest


_DATASET_ONLY_FIELDS = {"name", "output_dir"}


@dataclass
class DatasetSuiteConfig:
    seed: Optional[int]
    clean_output: bool
    datasets_root: str
    datasets: List[DatasetGenerateRequest] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    chain_configs: list[dict[str, Any]] = field(default_factory = list)


def _merge(shared: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge: any key present in `entry` replaces the shared value
    (lists are replaced as a whole, never element-wise)."""
    merged: Dict[str, Any] = {
        k: v for k, v in shared.items() if k not in _DATASET_ONLY_FIELDS
    }
    merged.update(entry)
    return merged


def load_dataset_config(path: str | Path) -> DatasetSuiteConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if "datasets" not in raw or not isinstance(raw["datasets"], list):
        raise ValueError("Config JSON must contain a 'datasets' list")

    shared = raw.get("shared", {}) or {}
    if not isinstance(shared, dict):
        raise ValueError("'shared' must be an object if present")

    chain_configs = []

    if 'chain_configs' in shared:
        chain_configs = shared['chain_configs']

    datasets_root = raw.get("datasets_root", "datasets")
    requests: List[DatasetGenerateRequest] = []
    names: List[str] = []

    for idx, entry in enumerate(raw["datasets"]):
        if not isinstance(entry, dict):
            raise ValueError(f"datasets[{idx}] must be an object")

        name = entry.get("name")
        if not name:
            raise ValueError(f"datasets[{idx}] is missing 'name'")

        merged = _merge(shared, entry)
        merged.pop("name", None)
        merged.setdefault("output_dir", f"{datasets_root}/{name}")

        try:
            request = DatasetGenerateRequest(**merged)
        except Exception as exc:
            raise ValueError(f"datasets[{idx}] ('{name}') is invalid: {exc}") from exc

        requests.append(request)
        names.append(name)

    return DatasetSuiteConfig(
        seed=raw.get("seed"),
        clean_output=bool(raw.get("clean_output", True)),
        datasets_root=datasets_root,
        datasets=requests,
        names=names,
        chain_configs = chain_configs
    )
