"""Request / response models for graph and labeling endpoints."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from web.backend.models.common import CytoscapeElement
from web.backend.models.distribution_models import IntDistributionModel


class MotifConfig(BaseModel):
    type: str = Field(..., description="Motif type, e.g. 'cycle', 'house', 'star'")
    params: List[Any] = Field(
        default_factory=list,
        description="Positional params for the motif generator, e.g. [4] for cycle",
    )
    count: int = Field(1, ge=1, description="Fixed number of instances of this motif")
    count_distribution: Optional[IntDistributionModel] = Field(
        None,
        description="Random distribution for the instance count (overrides count when present)",
    )

    def to_list(self) -> list:
        return [self.type] + list(self.params)


class ExtrasMixin(BaseModel):
    """Extra vertices / edges added after composition: a fixed count or a
    distribution sampled per graph (the distribution wins when present)."""

    num_extra_vertices: int = Field(0, ge=0, description="Fixed number of extra vertices")
    num_extra_vertices_distribution: Optional[IntDistributionModel] = Field(
        None,
        description="Random distribution for the extra-vertex count (overrides num_extra_vertices when present)",
    )
    num_extra_edges: int = Field(0, ge=0, description="Fixed number of extra edges")
    num_extra_edges_distribution: Optional[IntDistributionModel] = Field(
        None,
        description="Random distribution for the extra-edge count (overrides num_extra_edges when present)",
    )


class GraphGenerateRequest(ExtrasMixin):
    motifs: List[MotifConfig] = Field(..., min_length=1)
    composition: str = "sequential"
    composition_params: Dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None


class GraphStats(BaseModel):
    num_nodes: int
    num_edges: int
    motif_counts: Dict[str, int]
    directed: bool = False
    multigraph: bool = False
    kg: bool = False
    relation_counts: Dict[str, int] = Field(default_factory=dict)
    entity_type_counts: Dict[str, int] = Field(default_factory=dict)


class GraphGenerateResponse(BaseModel):
    graph_id: str
    elements: List[CytoscapeElement]
    stats: GraphStats


class LabelingFunctionConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LabelAssignRequest(BaseModel):
    graph_id: str
    labeling_functions: List[LabelingFunctionConfig] = Field(default_factory=list)


class LabelDistribution(BaseModel):
    counts: Dict[str, int]


class LabeledGraphResponse(BaseModel):
    graph_id: str
    elements: List[CytoscapeElement]
    label_distribution: LabelDistribution


class GraphUploadResponse(BaseModel):
    graph_id: str
    elements: List[CytoscapeElement]
    stats: GraphStats
    file_count: int
    folder_path: Optional[str] = None
    warnings: List[Dict[str, str]] = Field(default_factory=list)
