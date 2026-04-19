from typing import Iterator
from abc import ABC
from abc import abstractmethod

from app.models import Concept, Entity, Relation

class DataLoaderContract(ABC):

    @abstractmethod
    def load_dataset(self) -> list[(str, str)]:
        pass

    def load_entities(self) -> Iterator[Entity]:
        pass

    def load_concepts(self) -> Iterator[Concept]:
        pass

    def load_entity_isa_concept_relations(self) -> Iterator[Relation]:
        pass

    def load_entity_to_entity_relations(self) -> Iterator[Relation]:
        pass

    def load_concept_to_concept_relations(self) -> Iterator[Relation]:
        pass
