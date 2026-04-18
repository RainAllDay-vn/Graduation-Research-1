from abc import ABC
from abc import abstractmethod

from app.models import Concept, Entity

class DataLoaderContract(ABC):

    @abstractmethod
    def load_dataset(self) -> list[(str, str)]:
        pass

    def load_entities(self) -> list[Entity]:
        pass

    def load_concepts(self) -> list[Concept]:
        pass