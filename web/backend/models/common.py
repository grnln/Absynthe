"""Shared Pydantic models and Cytoscape serialisation helpers."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class NodeData(BaseModel):
    id: str
    label: str = "unknown"
    motif: str = ""
    motif_id: str = ""
    type: str = ""  # entity type (knowledge graphs)
    expected_ground_truth: str = ""
    observed_ground_truth: str = ""


class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    key: Optional[str] = None       # edge key (multigraphs only)
    relation: Optional[str] = None  # relation type (knowledge graphs)
    label: str = ""                 # edge-level label, when a labeler sets one


class CytoscapeElement(BaseModel):
    group: str  # "nodes" or "edges"
    data: Dict[str, Any]
    classes: str = ""


class CytoscapeGraph(BaseModel):
    elements: List[CytoscapeElement]
