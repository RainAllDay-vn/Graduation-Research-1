from typing import Iterator
from abc import ABC
from abc import abstractmethod

from app.models import Concept, Entity, Relation

class DataLoaderContract(ABC):

    @abstractmethod
    def load_dataset(self) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def load_entities(self) -> Iterator[Entity]:
        pass

    @abstractmethod
    def load_concepts(self) -> Iterator[Concept]:
        pass

    @abstractmethod
    def load_entity_isa_concept_relations(self) -> Iterator[Relation]:
        pass

    @abstractmethod
    def load_entity_to_entity_relations(self) -> Iterator[Relation]:
        pass

    @abstractmethod
    def load_concept_to_concept_relations(self) -> Iterator[Relation]:
        pass
