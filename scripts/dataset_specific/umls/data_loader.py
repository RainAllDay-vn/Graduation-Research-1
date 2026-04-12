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

    def _read_rrf(self, file_path: str, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
        """Helper to read RRF files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RRF file not found: {file_path}")
        
        if columns is None:
            columns = self._infer_columns(file_path)

        df = pd.read_csv(
            file_path, 
            sep='|', 
            names=columns, 
            index_col=False, 
            low_memory=False,
            skiprows=offset if offset else 0,
            nrows=limit if limit else None,
            chunksize=chunksize if chunksize else None,
            dtype=str
        )
        
        return df
    
    def _infer_columns(self, file_path: str) -> Optional[List[str]]:
        """Infer column names from MRFILES.RRF catalog."""
        try:
            df_files = self.load_file_definitions()
            df_relations = self.load_semantic_network_files()
            df = pd.concat([df_files, df_relations], ignore_index=True)
            filename = os.path.basename(file_path)
            file_info = df[df['FIL'] == filename]
            if not file_info.empty:
                fmt = file_info.iloc[0]['FMT']
                if fmt and isinstance(fmt, str):
                    return fmt.split(',')
        except Exception:
            pass
        return []

    def load_file_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRFILES.RRF which catalog all files in the release."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRFILES.RRF')
        columns = ['FIL', 'DES', 'FMT', 'CLS', 'RTY', 'SZY']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_column_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRCOLS.RRF which contains column-level metadata."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRCOLS.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load_mrdoc_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRDOC.RRF which contains metadata documentation (key-value maps)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRDOC.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load_source_vocabularies(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRSAB.RRF which contains source-level metadata (registry)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRSAB.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load_ranking_metadata(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRRANK.RRF which contains preferred term ranking within sources."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRRANK.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load_semantic_network_files(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads SRFIL which identifies all files in the Semantic Network (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRFIL')
        columns = ['FIL', 'DES', 'FMT', 'CLS', 'RTY', 'SZY']
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_semantic_network_fields(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads SRFLD which contains field descriptions for the Semantic Network (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRFLD')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load_semantic_network_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads SRDEF which contains the definitions of Semantic Types and Relations (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRDEF')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load_concepts(self, limit: Optional[int] = None, offset: Optional[int] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
        """Loads MRCONSO.RRF which contains every concept name (atom) in the Metathesaurus."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRCONSO.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset, chunksize=chunksize)

    def load_concept_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRDEF.RRF which contains semantic definitions for concepts."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRDEF.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load_semantic_types(self, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRSTY.RRF which maps each concept to its semantic type(s)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRSTY.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset)

    def load(self):
        """Loads UMLS data sequentially into the Neo4j database."""
        self._clear_database()
        self._create_constraints()
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

    def _create_constraints(self):
        print("Creating constraints...")
        query = '''
        CREATE CONSTRAINT id_unique IF NOT EXISTS
        FOR (b:Base) REQUIRE b.id IS UNIQUE;
        '''
        with self.driver.session() as session:
            session.run(query)

    def _insert_concepts(self):
        print("Inserting concepts...")

        concepts_map = {}
        progress = 0
        columns = self.load_concepts(limit=1).columns.tolist()
        cui_index = columns.index('CUI')
        ispref_index = columns.index('ISPREF')
        str_index = columns.index('STR')
        for df in self.load_concepts(chunksize=100_000):
            for row in df.itertuples():
                if row[cui_index] in concepts_map: continue
                if row[ispref_index] == 'N': continue
                concept = {}
                concept['id'] = row[cui_index]
                concept['name'] = row[str_index]
                concepts_map[concept['id']] = concept
            progress += len(df)
            print(f'Progress: {progress} rows')
            
        query = """
        UNWIND $batch as item
        MERGE (c:Concept:Base {id: item.id})
        SET c.name = item.name
        """
        with self.driver.session() as session:
            session.run(query, batch=concepts_map.values())

def get_loader(driver: Driver, dataset_path: str) -> UmlsDataLoader:
    return UmlsDataLoader(driver, dataset_path)