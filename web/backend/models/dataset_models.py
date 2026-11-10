"""Request / response models for dataset endpoints."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from graph.exporters import DEFAULT_FORMATS

from web.backend.models.graph_models import ExtrasMixin, LabelingFunctionConfig, MotifConfig
from web.backend.models.perturbation_models import PerturbationConfig


class FolderSourceConfig(BaseModel):
    folder_path: str
    iteration_order: str = "sequential"
    exhaustion_policy: str = "stop"
    as_undirected: bool = Field(
        False,
        description="Collapse loaded graphs onto simple undirected graphs (legacy behaviour)",
    )
    rdf_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Options for the RDF loader: directed, multigraph, type_predicates, short_names",
    )


class RelationSpec(BaseModel):
    name: str
    domain: str
    range: str
    p: Optional[float] = Field(None, ge=0.0, le=1.0)
    count: Optional[int] = Field(None, ge=0)


class KGSchemaConfig(BaseModel):
    """Schema for the synthetic knowledge-graph generator (``SchemaKGGenerator``)."""

    entity_types: Dict[str, int]
    relations: List[RelationSpec]
    directed: bool = True
    multigraph: bool = True
    allow_self_loops: bool = False


class DatasetGenerateRequest(ExtrasMixin):
    num_graphs: int = Field(..., ge=1, le=10000)
    motifs: Optional[List[MotifConfig]] = None
    folder_source: Optional[FolderSourceConfig] = None
    kg_schema: Optional[KGSchemaConfig] = None
    composition: str = "sequential"
    composition_params: Dict[str, Any] = Field(default_factory=dict)
    labeling_functions: List[LabelingFunctionConfig] = Field(default_factory=list)
    perturbations: List[PerturbationConfig] = Field(default_factory=list)
    max_perturbation_iterations: int = Field(10, ge=1, le=200)
    formats: List[str] = Field(
        default_factory=lambda: list(DEFAULT_FORMATS),
        description="Export formats (names registered in graph.exporters); the first one is the primary graph file",
    )
    output_dir: str = "datasets/output"
    seed: Optional[int] = None

    @model_validator(mode="after")
    def validate_graph_source(self):
        sources = [
            name for name, value in (
                ("motifs", self.motifs),
                ("folder_source", self.folder_source),
                ("kg_schema", self.kg_schema),
            )
            if value
        ]
        if len(sources) > 1:
            raise ValueError(
                "Specify only one graph source among 'motifs', 'folder_source' and 'kg_schema'."
            )
        if not sources:
            raise ValueError("Must specify one of 'motifs', 'folder_source' or 'kg_schema'.")
        return self


class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    current: int
    total: int
    output_dir: Optional[str] = None
    error: Optional[str] = None


class DatasetGraphRecord(BaseModel):
    graph_id: int
    graph_path: str
    perturbation_info: Optional[Dict[str, Any]]
    base_graph_id: Optional[int] = None
    original_graph_path: Optional[str] = None
    perturbation_name: Optional[str] = None


class DatasetListItem(BaseModel):
    output_dir: str
    num_graphs: int
