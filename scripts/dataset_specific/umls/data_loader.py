import os
import pandas as pd
from typing import Optional, List
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

    def _read_rrf(self, file_path: str, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Helper to read RRF files with pipe delimiters."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RRF file not found: {file_path}")
        
        # RRF files often have a trailing | which creates an extra empty column
        df = pd.read_csv(
            file_path, 
            sep='|', 
            names=columns, 
            index_col=False, 
            low_memory=False,
            skiprows=offset if offset else 0,
            nrows=limit if limit else None
        )
            
        return df

    def load_file_definitions(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRFILES.RRF which catalog all files in the release."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRFILES.RRF')
        if columns is None:
            columns = ['FIL', 'DES', 'FMT', 'CLS', 'RTY', 'SZY']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_column_definitions(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRCOLS.RRF which contains column-level metadata."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRCOLS.RRF')
        if columns is None:
            columns = ['COL', 'DES', 'REF', 'MIN', 'AV', 'MAX', 'FIL', 'DTY']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_mrdoc_definitions(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRDOC.RRF which contains metadata documentation (key-value maps)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRDOC.RRF')
        if columns is None:
            columns = ['DOCKEY', 'VALUE', 'TYPE', 'EXPL']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_source_vocabularies(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRSAB.RRF which contains source-level metadata (registry)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRSAB.RRF')
        if columns is None:
            columns = [
                'VCUI', 'RCUI', 'VSAB', 'RSAB', 'SON', 'SF', 'SVER', 'VSTART', 
                'VEND', 'IMETA', 'RMETA', 'SLC', 'SCC', 'SRL', 'TFR', 'CFR', 
                'CXTY', 'TTYL', 'ATNL', 'LAT', 'CENC', 'CURVER', 'SABIN', 'SSN', 'SCIT'
            ]
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_semantic_network_definitions(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads SRDEF which contains the definitions of Semantic Types and Relations (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRDEF')
        if columns is None:
            columns = ['RT', 'UI', 'NAME', 'TREE', 'DEF', 'EX', 'UN', 'NH', 'AB', 'RIN']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

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