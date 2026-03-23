from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Literal
from neo4j import Record

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
    type: Literal['Concept', 'Entity']
    name: str
    attributes: List[Attribute]
    relationships: list[Relationship]

    @staticmethod
    def from_database_node(node):
        if 'Concept' in node['labels']:
            type = 'Concept'
        elif 'Entity' in node['labels']:
            type = 'Entity'
        else:
            raise ValueError("The input node doesn't have a valid label")
        return Node(
            id=node['id'],
            type=type,
            name=node['name'],
            attributes=[],
            relationships=[]
        )
