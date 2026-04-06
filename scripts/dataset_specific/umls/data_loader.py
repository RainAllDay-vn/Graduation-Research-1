import os
import pandas as pd
from typing import Optional
from neo4j import Driver
from scripts.models import DataLoader

class UmlsDataLoader(DataLoader):
    def __init__(self, driver: Driver, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = os.path.join('dataset', 'umls')
        super().__init__(driver, dataset_path)

    def load(self):
        """Loads UMLS data sequentially into the Neo4j database."""
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Dataset path '{self.dataset_path}' is not a directory.")

        # Updated to use self.dataset_path
        extracted_path = os.path.join(self.dataset_path, 'extracted', '2025AB')

        self._clear_database()
        self._insert_concepts(extracted_path)

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

    def _insert_concepts(self, extracted_path: str):
        print("Inserting concepts...")
        headers = self.load_headers(extracted_path, 'MRCONSO.RRF')
        file_path = os.path.join(extracted_path, 'META', 'MRCONSO.RRF')
        reader = pd.read_csv(
            file_path, 
            sep='|',
            names=headers,
            index_col=False, 
            low_memory=False, 
            chunksize=500_000,
            usecols=range(len(headers))
        )

        for chunk in reader:
            chunk.info(memory_usage='deep')
            return

    def load_headers(self, extracted_path: str, file_name: str):
        print(f"Loading metadata for {file_name}...")
        files_description_path = os.path.join(extracted_path, 'META', 'MRFILES.RRF')
        files_description = pd.read_csv(
            files_description_path, 
            sep='|', 
            names=['FIL', 'DES', 'FMT', 'CLS', 'RWS', 'BTS'], 
            index_col=False, 
            usecols=range(6)
        )

        file_description = files_description[files_description['FIL'] == file_name]
        headers = file_description['FMT'].iloc[0].split(',')
        return headers

    def load_headers_with_description(self, extracted_path: str, file_name: str):
        print(f"Printing file info for {file_name}...")
        
        headers_description_path = os.path.join(extracted_path, 'META', 'MRCOLS.RRF')
        headers_description = pd.read_csv(
            headers_description_path, 
            sep='|', 
            names=['COL', 'DES', 'REF', 'MIN', 'AV', 'MAX', 'FIL', 'DTY'], 
            index_col=False, 
            usecols=range(8)
        )
        
        headers = self.load_headers(extracted_path, file_name)   
        for header in headers:
            header_info = headers_description[headers_description['COL'] == header]
            print(f'{header}: {header_info["DES"].iloc[0]}')

def get_loader(driver: Driver, dataset_path: str) -> UmlsDataLoader:
    return UmlsDataLoader(driver, dataset_path)