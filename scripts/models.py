from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Literal

@dataclass
class QualifierUnit:
    type: Literal['string', 'quantity', 'date', 'year']
    value: Union[float, int, str]
    unit: Optional[str] = None

@dataclass
class Qualifier:
    key: str
    values: List[QualifierUnit] = field(default_factory=list)

@dataclass
class Attribute:
    key: str
    value: QualifierUnit
    qualifiers: Dict[str, List[QualifierUnit]] = field(default_factory=dict)

@dataclass
class Relationship:
    predicate: str
    target: Node
    qualifiers: Dict[str, List[QualifierUnit]] = field(default_factory=dict)

@dataclass
class Node:
    id: str
    name: str
    attributes: List[Attribute]
    relationships: list[Relationship]
