from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Literal
from abc import ABC, abstractmethod
from neo4j import Record, Driver

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
class Relation:
    name: str
    target: "Concept | Entity"
    qualifiers: Dict[str, List[QualifierUnit]] = field(default_factory=dict)

@dataclass
class Concept:
    id: str
    name: str

@dataclass
class Entity:
    id: str
    name: str
    attributes: List[Attribute]
    relationships: list[Relation]

class DataLoader(ABC):
    def __init__(self, driver: Driver, dataset_path: Optional[str] = None):
        self.driver = driver
        self.dataset_path = dataset_path

    @abstractmethod
    def load(self):
        """Loads data into the Neo4j database."""
        pass
