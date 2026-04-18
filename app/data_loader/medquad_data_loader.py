import os
import pandas as pd
from typing import Optional
from app.data_loader.contract import DataLoaderContract

class MedquadDataLoader(DataLoaderContract):
    def __init__(self, 
        dataset_path: Optional[str] = None
    ):
        if dataset_path is None:
            dataset_path = os.path.join('dataset', 'medquad', 'test.csv')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' is not a file.")

        self.dataset_path = dataset_path

    def load_dataset(self) -> list[(str, str)]:
        data = pd.read_csv(self.dataset_path)

        dataset = []
        for _, row in data.iterrows():
            question = row['question']
            answer = row['answer']
            dataset.append((question, answer))

        return dataset
        