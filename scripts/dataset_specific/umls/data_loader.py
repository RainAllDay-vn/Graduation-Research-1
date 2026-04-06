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
        self._insert_concepts()

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
        pass

    def load_file(self, folder_name: str, file_name: str, offset: int = 0, limit: int = 100_000):
        headers = self.load_headers(file_name)
        file_path = os.path.join(self.extracted_path, folder_name, file_name)
        return pd.read_csv(
            file_path, 
            sep='|',
            names=headers,
            index_col=False, 
            low_memory=False, 
            skiprows=offset,
            nrows=limit,
            usecols=range(len(headers))
        )

    def load_headers(self, file_name: str):
        print(f"Loading metadata for {file_name}...")
        files_description_path = os.path.join(self.extracted_path, 'META', 'MRFILES.RRF')
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

    def load_headers_with_description(self, file_name: str):
        print(f"Printing file info for {file_name}...")
        
        headers_description_path = os.path.join(self.extracted_path, 'META', 'MRCOLS.RRF')
        headers_description = pd.read_csv(
            headers_description_path, 
            sep='|', 
            names=['COL', 'DES', 'REF', 'MIN', 'AV', 'MAX', 'FIL', 'DTY'], 
            index_col=False, 
            usecols=range(8)
        )
        headers_description = headers_description[headers_description['FIL'] == file_name]
        headers = self.load_headers(file_name)   

        result = []
        for header in headers:
            match = headers_description[headers_description['COL'] == header]
            description = match['DES'].iloc[0] if not match.empty else "No description available"
            result.append((header, description))
            
        return result

def get_loader(driver: Driver, dataset_path: str) -> UmlsDataLoader:
    return UmlsDataLoader(driver, dataset_path)