import os
import csv
import subprocess
from typing import Iterator, List, Optional

import pandas as pd
from app.data_loader.contract import DataLoaderContract
from app.models import Entity, Concept, Relation
from app.utils import to_screaming_snake_case

class UMLSDataLoader(DataLoaderContract):
    def __init__(self, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = os.path.join('dataset', 'umls')

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' is not a directory.")

        self.dataset_path = os.path.join(dataset_path, 'extracted', '2025AB')

    def load_concepts(self) -> Iterator[Concept]:
        columns = ['TUI', 'STY']
        tui_set: set[str] = set()
        for df in self.load_semantic_types(chunksize=1_000_000, columns=columns):
            for row in df.itertuples(index=False):
                tui, sty = row
                if tui in tui_set:
                    continue
                tui_set.add(tui)
                yield Concept(id=tui, name=sty, labels=[to_screaming_snake_case(sty)])

    def load_entities(self) -> Iterator[Entity]:
        columns = ['CUI', 'ISPREF', 'STR']
        cui_set: set[str] = set()
        for df in self.load_concept_names(chunksize=1_000_000, columns=columns):
            for row in df.itertuples(index=False):
                cui, ispref, name = row
                if cui in cui_set or ispref == 'N':
                    continue
                cui_set.add(cui)
                yield Entity(id=cui, name=name, labels=[])

    def load_entity_isa_concept_relations(self) -> Iterator[Relation]:
        cui_set: set[str] = set()
        for entity in self.load_entities():
            cui_set.add(entity.id)

        columns = ['CUI', 'TUI']
        for df in self.load_semantic_types(chunksize=1_000_000, columns=columns):
            for row in df.itertuples(index=False):
                cui, tui = row
                if cui not in cui_set:
                    continue
                yield Relation(source_id=cui, target_id=tui, label='ISA')

    def load_entity_to_entity_relations(self) -> Iterator[Relation]:
        temp_dir = os.path.join(self.dataset_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        raw_file = os.path.join(temp_dir, 'rels_raw.csv')
        sorted_file = os.path.join(temp_dir, 'rels_sorted.csv')

        inverse_map = {}
        mrdoc_df = self.load_mrdoc_definitions()
        mrdoc_df = mrdoc_df[(mrdoc_df['DOCKEY'] == 'REL') & (mrdoc_df['TYPE'] == 'rel_inverse')]
        mrdoc_df = mrdoc_df[['VALUE', 'EXPL']]
        for row in mrdoc_df.itertuples(index=False):
            inverse_map[row[0]] = row[1]

        print("Phase 1: Streaming relationships to disk...")
        with open(raw_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|')

            columns = ['CUI1', 'CUI2', 'STYPE1', 'STYPE2', 'REL', 'RELA']
            for df in self.load_relationships(chunksize=1_000_000, columns=columns):
                for row in df.itertuples(index=False):
                    cui1, cui2, stype1, stype2, rel, rela = row
                    if stype1 != 'SCUI' or stype2 != 'SCUI' or rela == '':
                        continue

                    writer.writerow([rel, cui1, cui2, rela])
                    if rel in inverse_map:
                        writer.writerow([inverse_map[rel], cui2, cui1, rela])

        print("Phase 2: Sorting relationships...")
        subprocess.run(['sort', '-t|', '-u', raw_file, '-o', sorted_file], check=True)

        print("Phase 3: Yielding relationships...")
        with open(sorted_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if not row:
                    continue
                rel, c1, c2, rela = row
                yield Relation(
                    source_id=c1,
                    target_id=c2,
                    label=to_screaming_snake_case(rel),
                    name=rela
                )

        print("Phase 4: Cleanup...")
        os.remove(raw_file)
        os.remove(sorted_file)

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

    def load_concept_names(self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        chunksize: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Loads MRCONSO.RRF which contains every concept name (atom) in the Metathesaurus."""
        file_path = os.path.join(self.dataset_path, 'META', 'MRCONSO.RRF')
        return self._read_rrf(
            file_path,
            limit=limit,
            offset=offset,
            chunksize=chunksize,
            columns=columns
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
