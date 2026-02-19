from dataclasses import dataclass, field
from typing import Dict, Tuple


GridPos = Tuple[int, int]
NODE_OPEN = "Op"
NODE_CLOSED = "Cl"


@dataclass
class RCGNode:
    node_id: int
    pos: GridPos
    node_type: str
    lap: int
    iteration: int
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class RCGEdge:
    u: int
    v: int
    weight: float
    edge_type: str
    tentative: bool = False
