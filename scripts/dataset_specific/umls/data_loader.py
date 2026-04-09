import os
import pandas as pd
from typing import Optional
from neo4j import Driver
from models import DataLoader

class UmlsDataLoader(DataLoader):
    def __init__(self, driver: Driver, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = os.path.join('dataset', 'umls')

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' is not a directory.")
        
        self.extracted_path = os.path.join(dataset_path, 'extracted', '2025AB')
        super().__init__(driver, dataset_path)

    def load(self):
        """Loads UMLS data sequentially into the Neo4j database."""
        self._clear_database()

    def _clear_database(self):
        print("Clearing database...")
        cleanup_query = """
        CALL apoc.periodic.iterate(
        "MATCH (n) RETURN n",
        "DETACH DELETE n",
        {batchSize: 2000, parallel: false}
        )
        """
        with self.driver.session() as session:
            session.run(cleanup_query)

def get_loader(driver: Driver, dataset_path: str) -> UmlsDataLoader:
    return UmlsDataLoader(driver, dataset_path)