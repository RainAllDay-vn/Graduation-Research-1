import os
from typing import Iterator, Optional

import pandas as pd

from app.data_loader.contract import DataLoaderContract
from app.models import Concept, Entity, Relation

class MedquadDataLoader(DataLoaderContract):
    def __init__(self, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = os.path.join('dataset', 'medquad', 'test.csv')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' is not a file.")

        self.dataset_path = dataset_path

    def load_dataset(self) -> list[tuple[str, str]]:
        data = pd.read_csv(self.dataset_path)

        dataset = []
        for _, row in data.iterrows():
            question = row['question']
            answer = row['answer']
            dataset.append((question, answer))

        return dataset

    def load_concepts(self) -> Iterator[Concept]:
        yield from []

    def load_entities(self) -> Iterator[Entity]:
        yield from []

    def load_entity_isa_concept_relations(self) -> Iterator[Relation]:
        yield from []

    def load_entity_to_entity_relations(self) -> Iterator[Relation]:
        yield from []

    def load_concept_to_concept_relations(self) -> Iterator[Relation]:
        yield from []
