"""Metadata files written next to the graphs of a generated dataset.

Two JSON files accompany every dataset:

``metadata.json``
    A list with one entry per perturbed variant: its paths, base graph and
    the reversible trace of the perturbation (see ``graph.reconstruction``).

``config.json``
    The provenance of the run, so the dataset can be audited or regenerated
    from its own folder::

        {
            "seed":       42,
            "config":     {... declarative config that produced the dataset ...},
            "components": {"graph_generator": ..., "labeling_functions": [...], "perturbations": [...]},
            "generation": {"num_graphs": 300, "graph_kwargs": {...}, ...},
            "provenance": {... optional: config file, position in a suite ...}
        }

    ``config`` is the declarative description (the JSON dataset entry / web
    request) when the dataset was generated from one, and ``components`` is
    always derived from the live objects, so a dataset built programmatically
    is documented too.
"""
from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

METADATA_FILENAME = "metadata.json"
CONFIG_FILENAME = "config.json"

# Constructor arguments that are runtime objects rather than configuration.
_NON_CONFIG_PARAMS = {"rng", "self"}


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of configuration objects to JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "model_dump"):  # pydantic models
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None) or repr(value)
        return f"{getattr(value, '__module__', '?')}.{name}"
    return str(value)


def describe_component(obj: Any) -> Dict[str, Any]:
    """Describe a pluggable component (generator, labeler, perturbation).

    Records the class and, for every constructor argument that the instance
    kept as an attribute of the same name, its current value. Components
    may override this by implementing ``describe() -> dict``.
    """
    custom = getattr(obj, "describe", None)
    if callable(custom):
        try:
            return to_jsonable(custom())
        except Exception:  # noqa: BLE001 - fall back to introspection
            pass

    cls = type(obj)
    desc: Dict[str, Any] = {"class": cls.__name__, "module": cls.__module__}
    params: Dict[str, Any] = {}
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for name, param in signature.parameters.items():
            if name in _NON_CONFIG_PARAMS or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if hasattr(obj, name):
                params[name] = to_jsonable(getattr(obj, name))
    desc["params"] = params
    folder = getattr(obj, "folder_name", None)
    if folder is not None:
        # Output subfolder of a perturbation (resolved default included).
        desc["folder_name"] = to_jsonable(folder)
    return desc


def describe_components(
    graph_generator: Any,
    labeling_functions: Iterable[Any],
    perturbations: Iterable[Tuple[Any, Any]],
) -> Dict[str, Any]:
    return {
        "graph_generator": describe_component(graph_generator),
        "labeling_functions": [describe_component(lf) for lf in labeling_functions],
        # ``count`` is an int, or the distribution dict when sampled per graph
        "perturbations": [
            {**describe_component(p), "count": to_jsonable(count)} for p, count in perturbations
        ],
    }


def build_config(
    *,
    seed: Optional[int],
    config: Optional[Dict[str, Any]],
    components: Dict[str, Any],
    generation: Dict[str, Any],
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the provenance document written as ``config.json``."""
    doc: Dict[str, Any] = {
        "seed": seed,
        "config": to_jsonable(config),
        "components": components,
        "generation": to_jsonable(generation),
    }
    if provenance:
        doc["provenance"] = to_jsonable(provenance)
    return doc


def _dump(data: Any, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def save_metadata(variants: List[Dict[str, Any]], path: str | os.PathLike) -> None:
    """Write the variants list as ``metadata.json``."""
    _dump(variants, path)


def save_config(doc: Dict[str, Any], path: str | os.PathLike) -> None:
    """Write the provenance document as ``config.json``."""
    _dump(doc, path)


def _resolve(path: str | os.PathLike, filename: str) -> Path:
    p = Path(path)
    return p / filename if p.is_dir() else p


def load_dataset_metadata(path: str | os.PathLike) -> List[Dict[str, Any]]:
    """Load ``metadata.json`` (or a dataset directory containing it): the variants list."""
    with open(_resolve(path, METADATA_FILENAME), encoding="utf-8") as f:
        variants = json.load(f)
    if not isinstance(variants, list):
        raise ValueError(f"{path} is not an Absynthe metadata.json (expected a list of variants)")
    return variants


def load_dataset_config(path: str | os.PathLike) -> Optional[Dict[str, Any]]:
    """Load ``config.json`` (or from a dataset directory); ``None`` when absent."""
    p = _resolve(path, CONFIG_FILENAME)
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_variant(variants: List[Dict[str, Any]], graph_id: int) -> Dict[str, Any]:
    """Return the variant entry with ``graph_id``."""
    for entry in variants:
        if entry.get("graph_id") == graph_id:
            return entry
    raise ValueError(f"graph_id {graph_id} not found in metadata")
