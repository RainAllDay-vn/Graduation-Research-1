import os
from typing import List, Optional

import pandas as pd
from app.data_loader.contract import DataLoaderContract
from app.models import Entity, Concept

class UMLSDataLoader(DataLoaderContract):
    def __init__(self, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = os.path.join('dataset', 'umls')

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' is not a directory.")

        self.dataset_path = os.path.join(dataset_path, 'extracted', '2025AB')

    def load_concepts(self) -> list[Concept]:
        pass

    def load_entities(self) -> list[Entity]:
        pass

    def _read_rrf(self,
        file_path: str,
        headings: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        chunksize: Optional[int] = None
    ):
        """Helper to read RRF files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RRF file not found: {file_path}")

        if headings is None:
            headings = self._infer_columns(file_path)

        # Normalize the headings to handle duplicates
        if headings:
            new_headings = []
            for i, col in enumerate(headings):
                if headings.count(col) > 1:
                    occurrence = headings[:i+1].count(col)
                    suffix = f"{occurrence}" if col == "STY/RL" else f"_{occurrence-1}"
                    new_headings.append(f"{col}{suffix}")
                else:
                    new_headings.append(col)
            headings = new_headings + ['']
        if columns is None:
            columns = headings

        result = pd.read_csv(
            file_path,
            sep='|',
            names=headings,
            usecols=columns,
            index_col=False,
            skiprows=offset if offset else 0,
            nrows=limit if limit else None,
            chunksize=chunksize if chunksize else None,
            dtype=str,
            na_filter=False,
            keep_default_na=False
        )

        return result

    def _infer_columns(self, file_path: str) -> Optional[List[str]]:
        """Infer column names from MRFILES.RRF catalog."""
        df_files = self.load_file_definitions()
        df_relations = self.load_semantic_network_files()
        df = pd.concat([df_files, df_relations], ignore_index=True)
        filename = os.path.basename(file_path)
        file_info = df[df['FIL'] == filename]
        if not file_info.empty:
            fmt = file_info.iloc[0]['FMT']
            if fmt and isinstance(fmt, str):
                return fmt.split(',')
        raise ValueError(f"Column names not found for file: {file_path}")

    def load_file_definitions(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRFILES.RRF which catalog all files in the release."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRFILES.RRF')
        headings = ['FIL', 'DES', 'FMT', 'CLS', 'RTY', 'SZY']
        return self._read_rrf(
            file_path,
            headings=headings,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_column_definitions(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRCOLS.RRF which contains column-level metadata."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRCOLS.RRF')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_mrdoc_definitions(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRDOC.RRF which contains metadata documentation (key-value maps)."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRDOC.RRF')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_source_vocabularies(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRSAB.RRF which contains source-level metadata (registry)."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRSAB.RRF')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_ranking_metadata(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRRANK.RRF which contains preferred term ranking within sources."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRRANK.RRF')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_semantic_network_files(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads SRFIL which identifies all files in the Semantic Network (NET directory)."""
        file_path = os.path.join(self.dataset_path, 'NET', 'SRFIL')
        headings = ['FIL', 'DES', 'FMT', 'CLS', 'RTY', 'SZY']
        return self._read_rrf(
            file_path,
            headings=headings,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_semantic_network_fields(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads SRFLD which contains field descriptions for the Semantic Network 
        (NET directory)."""
        file_path = os.path.join(self.dataset_path, 'NET', 'SRFLD')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_semantic_network_definitions(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads SRDEF which contains the definitions of Semantic Types and Relations 
        (NET directory)."""
        file_path = os.path.join(self.dataset_path, 'NET', 'SRDEF')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_semantic_network_relation_structure(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads SRSTR which contains the structure of the Semantic Network 
        (allowed relationships between types)."""
        file_path = os.path.join(self.dataset_path, 'NET', 'SRSTR')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_concept_definitions(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRDEF.RRF which contains semantic definitions for concepts."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRDEF.RRF')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_attributes(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRSAT.RRF which contains concept, term, and string attributes."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRSAT.RRF')
        return self._read_rrf(
            file_path,
            columns=columns,
            limit=limit,
            offset=offset
        )

    def load_semantic_types(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        chunksize: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRSTY.RRF which maps each concept to its semantic type(s)."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRSTY.RRF')
        return self._read_rrf(
            file_path,
            limit=limit,
            offset=offset,
            chunksize=chunksize,
            columns=columns
        )

    def load_hierarchies(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        chunksize: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRHIER.RRF which contains computable hierarchies."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRHIER.RRF')
        return self._read_rrf(
            file_path,
            limit=limit,
            offset=offset,
            chunksize=chunksize,
            columns=columns
        )

    def load_relationships(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        chunksize: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRREL.RRF which contains relationships between concepts."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRREL.RRF')
        return self._read_rrf(
            file_path,
            limit=limit,
            offset=offset,
            chunksize=chunksize,
            columns=columns
        )
