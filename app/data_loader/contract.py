from abc import ABC
from abc import abstractmethod

class DataLoaderContract(ABC):
    @abstractmethod
    def load_knowledge_graph(self) -> None:
        pass

    @abstractmethod
    def load_dataset(self) -> list[(str, str)]:
        pass
