"""Dataset generation service with progress tracking."""
import web.backend.services  # noqa: F401

import os
import threading
import uuid
from typing import Dict, Optional

from graph.dataset_generator import GraphDatasetGenerator
from web.backend.models.dataset_models import DatasetGenerateRequest, TaskStatus
from web.backend.services.registry import (
    build_graph_generator,
    build_graph_kwargs,
    build_labeling_functions,
    build_perturbations,
)

_tasks: Dict[str, dict] = {}
_lock = threading.Lock()


def create_task(total: int) -> str:
    task_id = str(uuid.uuid4())
    with _lock:
        _tasks[task_id] = {
            "status": "pending",
            "current": 0,
            "total": total,
            "output_dir": None,
            "error": None,
        }
    return task_id


def _update(task_id: str, **kwargs) -> None:
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].update(kwargs)


def get_status(task_id: str) -> Optional[TaskStatus]:
    with _lock:
        raw = _tasks.get(task_id)
    if raw is None:
        return None
    return TaskStatus(
        task_id=task_id,
        status=raw["status"],
        current=raw["current"],
        total=raw["total"],
        output_dir=raw.get("output_dir"),
        error=raw.get("error"),
    )


def run_generation(task_id: str, request: DatasetGenerateRequest) -> None:
    """Synchronous generation worker called from FastAPI BackgroundTasks."""
    from utils.rng import set_seed, reset_rng

    _update(task_id, status="running", current=0)

    if request.seed is not None:
        set_seed(request.seed)

    try:
        output_dir = os.path.abspath(request.output_dir)

        graph_generator = build_graph_generator(request)
        labeling_functions = build_labeling_functions(request.labeling_functions)
        perturbations = build_perturbations(request.perturbations)

        generator = GraphDatasetGenerator(
            graph_generator=graph_generator,
            labeling_functions=labeling_functions,
            perturbations=perturbations,
            output_dir=output_dir,
            max_perturbation_iterations=request.max_perturbation_iterations,
            formats=request.formats,
            seed=request.seed,
            config=request.model_dump(mode="json"),
            provenance={"source": "web", "task_id": task_id},
        )
        generator.generate_dataset(
            num_graphs=request.num_graphs,
            progress_callback=lambda done, _total: _update(task_id, current=done),
            **build_graph_kwargs(request),
        )

        _update(
            task_id,
            status="completed",
            current=request.num_graphs,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        _update(task_id, status="failed", error=str(exc))
    finally:
        if request.seed is not None:
            reset_rng()
