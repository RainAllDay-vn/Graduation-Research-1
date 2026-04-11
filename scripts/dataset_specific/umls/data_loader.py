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

    def convert_to_parquet(self, file_path: str, force: bool = False, columns: Optional[List[str]] = None):
        """Explicitly converts an RRF file to Parquet format for faster loading."""
        if file_path.endswith('.RRF'):
            parquet_path = file_path.replace('.RRF', '.parquet')
        else:
            parquet_path = file_path + '.parquet'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source RRF file not found: {file_path}")
            
        if os.path.exists(parquet_path) and not force:
            print(f"Parquet file already exists: {parquet_path}. Use force=True to overwrite.")
            return

        print(f"Converting {file_path} to Parquet...")
        
        # Attempt to auto-detect columns from metadata catalogs if not provided
        if columns is None:
            filename = os.path.basename(file_path)
            # 1. Try META directory catalog (MRFILES.RRF)
            try:
                df_files = self.load_file_definitions()
                file_info = df_files[df_files['FIL'] == filename]
                if not file_info.empty:
                    columns = file_info.iloc[0]['FMT'].split(',')
                    print(f"Detected columns from MRFILES.RRF: {columns}")
            except Exception:
                pass
            
            # 2. Try NET directory catalog (SRFIL) as fallback
            if columns is None:
                try:
                    df_net = self.load_semantic_network_files()
                    file_info = df_net[df_net['FIL'] == filename]
                    if not file_info.empty:
                        columns = file_info.iloc[0]['FMT'].split(',')
                        print(f"Detected columns from SRFIL: {columns}")
                except Exception:
                    pass

        # Use low_memory=False for large files to avoid type inference issues
        df = self._read_rrf(file_path, columns=columns, limit=None)
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"Saved to: {parquet_path}")


    def _read_rrf(self, file_path: str, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Helper to read files, preferring Parquet if available."""
        if file_path.endswith('.RRF'):
            parquet_path = file_path.replace('.RRF', '.parquet')
        else:
            parquet_path = file_path + '.parquet'
        
        # Prefer Parquet if it exists and we're not using limit/offset (which are tricky with Parquet unless we use Polars/PyArrow directly)
        if os.path.exists(parquet_path) and limit is None and offset is None:
            # print(f"Loading from Parquet: {parquet_path}")
            return pd.read_parquet(parquet_path)

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

    def load_ranking_metadata(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRRANK.RRF which contains preferred term ranking within sources."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRRANK.RRF')
        if columns is None:
            columns = ['RANK', 'SAB', 'TTY', 'SUPPRESS']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_semantic_network_files(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads SRFIL which identifies all files in the Semantic Network (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRFIL')
        if columns is None:
            columns = ['FIL', 'DES', 'FMT', 'CLS', 'RWS', 'BTS']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_semantic_network_fields(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads SRFLD which contains field descriptions for the Semantic Network (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRFLD')
        if columns is None:
            columns = ['COL', 'DES', 'REF', 'FIL']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)


    def load_semantic_network_definitions(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads SRDEF which contains the definitions of Semantic Types and Relations (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRDEF')
        if columns is None:
            columns = ['RT', 'UI', 'NAME', 'TREE', 'DEF', 'EX', 'UN', 'NH', 'AB', 'RIN']
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_concepts(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRCONSO.RRF which contains every concept name (atom) in the Metathesaurus."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRCONSO.RRF')
        if columns is None:
            columns = [
                'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 
                'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
            ]
        return self._read_rrf(file_path, columns, limit=limit, offset=offset)

    def load_concept_definitions(self, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """Loads MRDEF.RRF which contains semantic definitions for concepts."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRDEF.RRF')
        if columns is None:
            columns = ['CUI', 'AUI', 'ATUI', 'SATUI', 'SAB', 'DEF', 'SUPPRESS', 'CVF']
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