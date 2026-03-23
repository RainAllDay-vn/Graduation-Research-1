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
    name: str
    target: Node
    qualifiers: Dict[str, List[QualifierUnit]] = field(default_factory=dict)

    def __str__(self):
        return f'Relationship{{Name: {self.name}}}'

@dataclass
class Node:
    id: str
    type: Literal['Concept', 'Entity']
    name: str
    concept_node: Node | None
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
            concept_node=None,
            attributes=[],
            relationships=[]
        )
    
    def get_dfs_traversal(self)-> str:
        return self._dfs_traverse('', set())
    
    def _dfs_traverse(self, current_str: str, node_id_set: set[str]) -> str:
        if self.id in node_id_set:
            return current_str
        node_id_set.add(self.id)
        if self.concept_node != None:
            current_str += str(self) + '\n'
            current_str += str(self) + ' is a ' + str(self.concept_node) + '\n'
        for relationship in self.relationships:
            current_str += f'Node{str(self)} Relationship{str(relationship)} Node{str(relationship.target)}\n'
            relationship.target._dfs_traverse(current_str, node_id_set)
        return current_str    

    def __str__(self):
        return f'Node{{Type: {self.type} | Name:{self.name}}}'
