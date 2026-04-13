import os
import pandas as pd
import numpy as np
import subprocess
import csv
from typing import Optional, List
from neo4j import Driver
from models import DataLoader
from utils import to_screaming_snake_case

class UmlsDataLoader(DataLoader):
    def __init__(self, driver: Driver, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = os.path.join('dataset', 'umls')

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' is not a directory.")
        
        self.extracted_path = os.path.join(dataset_path, 'extracted', '2025AB')
        super().__init__(driver, dataset_path)

    def _read_rrf(self, file_path: str, schema: Optional[List[str]] = None, columns: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None, chunksize: Optional[int] = None):
        """Helper to read RRF files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RRF file not found: {file_path}")
        
        if schema is None:
            schema = self._infer_columns(file_path)

        actual_columns = list(schema) + [''] if schema else None
        usecols = columns if columns else (schema if schema else None)

        parquet_path = f"{file_path}.parquet"
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        target_is_large = file_size_mb > 100 or chunksize is not None

        if target_is_large and limit is None and offset is None:
            if not os.path.exists(parquet_path):
                print(f"Converting {os.path.basename(file_path)} to parquet (this only happens once)...")
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                csv_iter = pd.read_csv(
                    file_path, 
                    sep='|', 
                    names=actual_columns, 
                    usecols=schema if schema else None,
                    index_col=False, 
                    low_memory=False,
                    chunksize=1_000_000,
                    dtype=str,
                    na_filter=False,
                    keep_default_na=False
                )
                writer = None
                for chunk in csv_iter:
                    table = pa.Table.from_pandas(chunk)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema)
                    writer.write_table(table)
                if writer:
                    writer.close()
                print(f"Finished parquet conversion for {os.path.basename(file_path)}.")

            if chunksize:
                import pyarrow.parquet as pq
                def parquet_iterator():
                    pf = pq.ParquetFile(parquet_path)
                    for batch in pf.iter_batches(batch_size=chunksize, columns=columns):
                        yield batch.to_pandas()
                return parquet_iterator()
            else:
                return pd.read_parquet(parquet_path, columns=columns)

        df = pd.read_csv(
            file_path, 
            sep='|', 
            names=actual_columns, 
            usecols=usecols,
            index_col=False, 
            low_memory=False,
            skiprows=offset if offset else 0,
            nrows=limit if limit else None,
            chunksize=chunksize if chunksize else None,
            dtype=str,
            na_filter=False,
            keep_default_na=False
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

    def load_file_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRFILES.RRF which catalog all files in the release."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRFILES.RRF')
        file_schema = ['FIL', 'DES', 'FMT', 'CLS', 'RTY', 'SZY']
        return self._read_rrf(file_path, schema=file_schema, columns=columns, limit=limit, offset=offset)

    def load_column_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRCOLS.RRF which contains column-level metadata."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRCOLS.RRF')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_mrdoc_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRDOC.RRF which contains metadata documentation (key-value maps)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRDOC.RRF')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_source_vocabularies(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRSAB.RRF which contains source-level metadata (registry)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRSAB.RRF')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_ranking_metadata(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRRANK.RRF which contains preferred term ranking within sources."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRRANK.RRF')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_semantic_network_files(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads SRFIL which identifies all files in the Semantic Network (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRFIL')
        file_schema = ['FIL', 'DES', 'FMT', 'CLS', 'RTY', 'SZY']
        return self._read_rrf(file_path, schema=file_schema, columns=columns, limit=limit, offset=offset)

    def load_semantic_network_fields(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads SRFLD which contains field descriptions for the Semantic Network (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRFLD')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_semantic_network_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads SRDEF which contains the definitions of Semantic Types and Relations (NET directory)."""
        file_path = os.path.join(self.extracted_path, 'NET', 'SRDEF')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_concepts(self, limit: Optional[int] = None, offset: Optional[int] = None, chunksize: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRCONSO.RRF which contains every concept name (atom) in the Metathesaurus."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRCONSO.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset, chunksize=chunksize, columns=columns)

    def load_concept_definitions(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRDEF.RRF which contains semantic definitions for concepts."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRDEF.RRF')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_attributes(self, limit: Optional[int] = None, offset: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRSAT.RRF which contains concept, term, and string attributes."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRSAT.RRF')
        return self._read_rrf(file_path, columns=columns, limit=limit, offset=offset)

    def load_semantic_types(self, limit: Optional[int] = None, offset: Optional[int] = None, chunksize: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRSTY.RRF which maps each concept to its semantic type(s)."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRSTY.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset, chunksize=chunksize, columns=columns)

    def load_hierarchies(self, limit: Optional[int] = None, offset: Optional[int] = None, chunksize: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRHIER.RRF which contains computable hierarchies."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRHIER.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset, chunksize=chunksize, columns=columns)

    def load_relationships(self, limit: Optional[int] = None, offset: Optional[int] = None, chunksize: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads MRREL.RRF which contains relationships between concepts."""
        file_path = os.path.join(self.extracted_path, 'META', 'MRREL.RRF')
        return self._read_rrf(file_path, limit=limit, offset=offset, chunksize=chunksize, columns=columns)

    def load(self):
        """Loads UMLS data sequentially into the Neo4j database."""
        self._clear_database()
        self._create_constraints()
        self._insert_entities()
        self._insert_concepts()
        self._insert_entity_concept_relations()
        self._insert_entity_to_entity_relations()

    def _clear_database(self):
        print("Clearing database...")
        # Clear data
        cleanup_query = """
        CALL apoc.periodic.iterate(
        "MATCH (n) RETURN n",
        "DETACH DELETE n",
        {batchSize: 2000, parallel: false}
        )
        """
        with self.driver.session() as session:
            session.run(cleanup_query).consume()

        # Clear schema (constraints and indexes)
        cleanup_query = "CALL apoc.schema.assert({}, {})"
        with self.driver.session() as session:
            session.run(cleanup_query).consume()

        # Verify emptiness
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            
            if node_count > 0 or rel_count > 0:
                print(f"WARNING: Database not fully cleared. Nodes: {node_count}, Rels: {rel_count}")
                print("Retrying simple clear...")
                session.run("MATCH (n) DETACH DELETE n").consume()
            else:
                print("Database cleared successfully.")

    def _create_constraints(self):
        print("Creating constraints...")
        query = '''
        CREATE CONSTRAINT id_unique IF NOT EXISTS
        FOR (b:Base) REQUIRE b.id IS UNIQUE;
        '''
        with self.driver.session() as session:
            session.run(query)

    def _insert_entities(self):
        print("Inserting entities...")

        entities_map = {}
        progress = 0
        target_cols = ['CUI', 'ISPREF', 'STR']
        for df in self.load_concepts(chunksize=1_000_000, columns=target_cols):
            for row in df.itertuples(index=False):
                cui = row[0] # CUI
                ispref = row[1] # ISPREF
                name = row[2] # STR

                if cui in entities_map or ispref == 'N':
                    continue

                entities_map[cui] = {
                    'id': cui,
                    'name': name
                }
            progress += len(df)
            print(f'Progress: {progress} rows')
            
        query = """
        UNWIND $batch as item
        CREATE (c:Entity:Base {id: item.id, name: item.name})
        """
        self._insert_batch(list(entities_map.values()), query)

    def _insert_concepts(self):
        print("Inserting concepts...")

        concepts_map = {}
        progress = 0
        target_cols = ['TUI', 'STY']
        for df in self.load_semantic_types(chunksize=1_000_000, columns=target_cols):
            for row in df.itertuples(index=False):
                tui = row[0] # TUI
                sty = row[1] # STY
                if tui in concepts_map: continue
                concept = {
                    'id': tui,
                    'name': sty
                }
                concepts_map[tui] = concept
            progress += len(df)
            print(f'Progress: {progress} rows')
            
        query = """
        UNWIND $batch as item
        CREATE (c:Concept:Base {id: item.id, name: item.name})
        """
        self._insert_batch(list(concepts_map.values()), query)

    def _insert_entity_concept_relations(self):
        print("Inserting entity IS_A relations...")

        relation_map = set()
        progress = 0
        target_cols = ['CUI', 'TUI']
        for df in self.load_semantic_types(chunksize=1_000_000, columns=target_cols):
            for row in df.itertuples(index=False): 
                cui = row[0] # CUI
                tui = row[1] # TUI
                if (cui, tui) in relation_map: continue
                relation_map.add((cui, tui))
            progress += len(df)
            print(f'Progress: {progress} rows')
                
        data = [{'child_id': cui, 'parent_id': tui} for cui, tui in relation_map]
                
        query = """
        UNWIND $batch as item
        MATCH (child:Base {id: item.child_id})
        MATCH (parent:Base {id: item.parent_id})
        CREATE (child)-[:IS_A]->(parent)
        """
        self._insert_batch(data, query)

    def _insert_entity_to_entity_relations(self, limit: Optional[int] = None, keep_temp: bool = False):
        print("Inserting entity-to-entity relationships...")

        temp_dir = os.path.join(self.dataset_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        raw_file = os.path.join(temp_dir, 'rels_raw.csv')
        sorted_file = os.path.join(temp_dir, 'rels_sorted.csv')

        inverse_map = {}
        mrdoc_df = self.load_mrdoc_definitions()
        for row in mrdoc_df[(mrdoc_df['DOCKEY'] == 'REL') & (mrdoc_df['TYPE'] == 'rel_inverse')][['VALUE', 'EXPL']].itertuples(index=False): 
            inverse_map[row[0]] = row[1]

        print("Phase 1: Streaming relationships to disk...")
        with open(raw_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            
            def write_if_valid(r, c1, c2):
                if r and r.strip():
                    writer.writerow([r, c1, c2])

            progress = 0
            target_cols = ['CUI1', 'CUI2', 'STYPE1', 'STYPE2', 'REL']
            for df in self.load_relationships(chunksize=1_000_000, columns=target_cols, limit=limit):
                for row in df.itertuples(index=False): 
                    cui1, cui2, stype1, stype2, rel = row

                    if stype1 != 'SCUI' or stype2 != 'SCUI':
                        continue
                    
                    write_if_valid(rel, cui1, cui2)
                    if rel in inverse_map:
                        write_if_valid(inverse_map[rel], cui2, cui1)

                progress += len(df)
                print(f'Progress: {progress} rows')

        print("Phase 2: Sorting relationships externally...")
        # Sort and unique based on the whole line to group entries by the first column (REL)
        subprocess.run(['sort', '-t|', '-u', raw_file, '-o', sorted_file], check=True)

        print("Phase 3: Grouped insertion into Neo4j...")
        with open(sorted_file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            current_rel = None
            batch_data = []
            
            for row in reader:
                if not row: continue
                rel, c1, c2 = row
                
                if rel != current_rel:
                    if current_rel is not None:
                        self._process_rel_group(current_rel, batch_data)
                    current_rel = rel
                    batch_data = []
                
                batch_data.append((c1, c2))
                if len(batch_data) >= 100_000:
                    self._process_rel_group(current_rel, batch_data)
                    batch_data = []
            
            if current_rel is not None:
                self._process_rel_group(current_rel, batch_data)

        # Cleanup
        if not keep_temp:
            if os.path.exists(raw_file): os.remove(raw_file)
            if os.path.exists(sorted_file): os.remove(sorted_file)
            print("Temp files removed.")
        else:
            print(f"Temp files kept at: {temp_dir}")

        print("Entity-to-entity relationships insertion completed.")

    def _process_rel_group(self, rel: str, data: list):
        query = f"""
        UNWIND $batch as item
        MATCH (child:Base {{id: item[0]}})
        MATCH (parent:Base {{id: item[1]}})
        CREATE (child)-[:{to_screaming_snake_case(rel)}]->(parent)
        """
        self._insert_batch(data, query)

    def _insert_batch(self, data: list, query: str):
        print(f"Starting to insert {len(data)} rows...")

        chunksize = 25_000
        progress = 0
        
        with self.driver.session() as session:
            for i in range(0, len(data), chunksize):
                batch = data[i:i+chunksize]
                session.run(query, batch=batch)
                progress += len(batch)
                print(f'Inserting: {progress} rows')

def get_loader(driver: Driver, dataset_path: str = 'dataset/umls') -> UmlsDataLoader:
    return UmlsDataLoader(driver, dataset_path)