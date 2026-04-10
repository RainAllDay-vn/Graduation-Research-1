import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from typing import Dict, Any, List, Optional
    from neo4j import Driver
    from models import DataLoader
    from knowledge_graph import KnowledgeGraph
    from dataset_specific.umls.data_loader import UmlsDataLoader

    return KnowledgeGraph, UmlsDataLoader, mo, os


@app.cell
def _(mo):
    mo.md(r"""
    # Section 0: Setting up

    This notebook performs a comprehensive analysis of the **UMLS (Unified Medical Language System)** dataset.
    The goal is to understand the structure, scale, and content of the Metathesaurus files before processing or integrating them into the Knowledge Graph.
    """)
    return


@app.cell
def _(KnowledgeGraph, UmlsDataLoader, os):
    # Base configuration
    DATASET_ROOT = os.path.join("dataset", "umls")
    _kg = KnowledgeGraph()
    data_loader = UmlsDataLoader(_kg.driver, dataset_path=DATASET_ROOT)
    return (data_loader,)


@app.cell
def _(mo):
    mo.md(r"""
    # Section 1: Meta Datafiles (Reference Materials)

    In this section, we start exploring the **Meta Datafiles** (.RRF files that don't contain actual concept data, but rather documentation, structure, and definitions).
    These files are essential for understanding how the rest of the dataset is organized and how to interpret specific codes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.1 MRFILES.RRF (Master Catalog)

    **MRFILES.RRF** serves as the directory for all physical files included in the UMLS release. It provides metadata about file paths, descriptions, and record counts.

    ### Column Definitions:
    - **FIL**: Physical file name.
    - **DES**: Description of the file.
    - **FMT**: List of columns in the file (comma-separated).
    - **CLS**: Number of columns.
    - **RTY**: Total number of rows (record count).
    - **SZY**: Size in bytes.
    """)
    return


@app.cell
def _(data_loader):
    mrfiles_df = data_loader.load_file_definitions()
    return (mrfiles_df,)


@app.cell
def _(mrfiles_df):
    mrfiles_df.head(10)
    return


@app.cell
def _(mo, mrfiles_df):
    _mask = mrfiles_df['FIL'] == 'AMBIGLUI.RRF'
    _sample_row = mrfiles_df[_mask].iloc[0] if _mask.any() else mrfiles_df.iloc[0]

    mo.md(f"""
    ### 🔍 Row Interpretation: `{_sample_row['FIL']}`
    Let's break down the entry for the Ambiguous LUI identifiers file:

    - **File Name (`FIL`)**: `{_sample_row['FIL']}` (The physical file on disk)
    - **Description (`DES`)**: *{_sample_row['DES']}*
    - **Columns (`FMT`)**: `{_sample_row['FMT']}` (It contains {_sample_row['CLS']} columns: {_sample_row['FMT']})
    - **Scale**: It contains **{int(_sample_row['RTY']):,}** rows and occupies **{int(_sample_row['SZY']):,}** bytes.
    """)
    return


@app.cell
def _(mo, mrfiles_df):
    total_files = len(mrfiles_df)
    total_records = mrfiles_df['RTY'].sum()

    mo.md(f"""
    ### 📊 Dataset Summary
    - **Total Files**: {total_files}
    - **Total Records (All Files)**: {int(total_records):,}

    #### Top 10 Largest Files (by Record Count)
    """)
    return


@app.cell
def _(mrfiles_df):
    mrfiles_df.sort_values(by='RTY', ascending=False).head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.2 MRCOLS.RRF (Column Definitions)

    **MRCOLS.RRF** is the central data dictionary for the entire UMLS dataset. It defines every attribute (column) used across all files, providing descriptions, data types, and basic statistics.

    ### Key Attributes:
    - **COL**: Column name/Attribute name.
    - **DES**: Human-readable description.
    - **REF**: Reference to documentation.
    - **MIN/AV/MAX**: Statistics about the field length or value.
    - **FIL**: The file this column belongs to.
    - **DTY**: Data type.
    """)
    return


@app.cell
def _(data_loader):
    mrcols_df = data_loader.load_column_definitions()
    return (mrcols_df,)


@app.cell
def _(mrcols_df):
    mrcols_df.head(10)
    return


@app.cell
def _(mo, mrcols_df):
    # Let's find a descriptive column for demonstration
    _mask = mrcols_df['COL'] == 'CUI'
    _sample_col = mrcols_df[_mask].iloc[0] if _mask.any() else mrcols_df.iloc[0]

    mo.md(f"""
    ### 🔍 Column Interpretation: `{_sample_col['COL']}` in `{_sample_col['FIL']}`
    - **Description**: {_sample_col['DES']}
    - **Data Type**: `{_sample_col['DTY']}`
    - **Length Stats**: Min: {_sample_col['MIN']}, Avg: {_sample_col['AV']}, Max: {_sample_col['MAX']}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📂 Column Lookup by File
    Use the table below to explore columns for a specific file (e.g., `MRCONSO.RRF`).
    """)
    return


@app.cell
def _(mrcols_df):
    # Filter for MRCONSO.RRF as a common example
    mrcols_df[mrcols_df['FIL'] == 'MRCONSO.RRF']
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.3 MRDOC.RRF (Metadata Documentation)

    **MRDOC.RRF** is a key-value documentation map that explains codes, abbreviations, and attributes used throughout UMLS.

    ### Content Structure:
    - **DOCKEY**: The category of metadata (e.g., `ATN` for Attributes, `RELA` for Relationships).
    - **VALUE**: The specific code or abbreviation (e.g., `isa`).
    - **TYPE**: Category of explanation (usually `expanded_form`).
    - **EXPL**: The full description or definition.
    """)
    return


@app.cell
def _(data_loader):
    mrdoc_df = data_loader.load_mrdoc_definitions()
    return (mrdoc_df,)


@app.cell
def _(mo, mrdoc_df):
    dockey_counts = mrdoc_df['DOCKEY'].value_counts()

    mo.md(f"""
    ### 📊 Dataset Overview
    The documentation catalog contains **{len(dockey_counts)}** distinct categories. Below is the master view of the documentation map and the most frequent documentation types (**DOCKEY**):
    """)
    return (dockey_counts,)


@app.cell
def _(dockey_counts, mo, mrdoc_df):
    # Side-by-side or sequential overview
    mo.vstack([
        mo.md("#### Sample Entries"),
        mrdoc_df.head(5),
        mo.md("#### Frequencies"),
        dockey_counts.head(10)
    ])
    return


@app.cell
def _(mo, mrcols_df, mrdoc_df):
    # Simplified lookup logic
    def get_info(key):
        desc = mrcols_df[mrcols_df['COL'] == key]['DES'].iloc[0] if key in mrcols_df['COL'].values else "Metadata Category"
        data = mrdoc_df[mrdoc_df['DOCKEY'] == key]
        return desc, data

    atn_desc, atn_view = get_info('ATN')
    rela_desc, rela_view = get_info('RELA')

    mo.md(f"""
    ## 🔍 Deep Dive: Common Metadata Keys
    By linking **MRDOC** to **MRCOLS**, we can interpret high-volume documentation types:

    | Key | Meaning (from MRCOLS) | Count | Purpose |
    |-----|-------------------|-------|---------|
    | **ATN** | *{atn_desc}* | {len(atn_view)} | Maps technical codes to readable attributes. |
    | **RELA** | *{rela_desc}* | {len(rela_view):,} | Provides granular semantic context for relationships (e.g., `isa`, `part_of`). |
    """)
    return atn_view, rela_view


@app.cell
def _(atn_view, mo, rela_view):
    mo.vstack([
        mo.md("### 📂 Interpretation Previews"),
        mo.md("**Attributes (ATN):**"),
        atn_view.head(5),
        mo.md("**Relationship Labels (RELA):**"),
        rela_view.head(5)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.4 MRSAB.RRF (Source Vocabulary Registry)

    **MRSAB.RRF** contains detailed metadata about the external vocabularies (e.g., SNOMED CT, MeSH, ICD-10) that contribute concepts to the UMLS. It is the "Source Registry" that defines the provenance and versioning of the data.

    ### Content Highlights:
    - **SAB/VSAB/RSAB**: Source abbreviations (e.g., `MSH`, `SNOMEDCT_US`).
    - **SON/SF**: Official names and source families.
    - **SVER/VSTART/VEND**: Versioning and validity dates.
    - **LAT/CENC**: Language and character encoding.
    - **CFR/TFR**: Statistics (CUI and Term frequencies).
    """)
    return


@app.cell
def _(data_loader):
    mrsab_df = data_loader.load_source_vocabularies()
    return (mrsab_df,)


@app.cell
def _(mrsab_df):
    mrsab_df.head(10)
    return


@app.cell
def _(mo, mrsab_df):
    lang_counts = mrsab_df['LAT'].value_counts()

    mo.md(f"""
    ### 🌍 Global Reach
    The current release includes contents in **{len(lang_counts)}** languages.

    #### Sources by Language (Top 10)
    """)
    return (lang_counts,)


@app.cell
def _(lang_counts):
    lang_counts.head(10)
    return


@app.cell
def _(mo, mrsab_df):
    # Example: Look for a major source
    _major_source = mrsab_df[mrsab_df['RSAB'] == 'MSH'].iloc[0] if 'MSH' in mrsab_df['RSAB'].values else mrsab_df.iloc[0]

    mo.md(f"""
    ### 🔍 Source Detailed Preview: `{_major_source['RSAB']}`
    - **Official Name**: {_major_source['SON']}
    - **Family**: {_major_source['SF'] or 'N/A'}
    - **Version**: `{_major_source['SVER']}`
    - **Language**: {_major_source['LAT']}
    - **Scale**: Contains **{int(_major_source['CFR']):,}** concepts (CUIs).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.5 MRRANK.RRF (Concept Name Ranking)

    **MRRANK.RRF** contains metadata about how concept names (terms) are ranked within each source vocabulary. This ranking determines which term is considered the "Preferred Name" (PN) for a concept.

    ### Key Column Definitions:
    - **RANK**: Absolute numerical rank (higher is better).
    - **SAB**: Source abbreviation (e.g., `MTH`, `SNOMEDCT_US`).
    - **TTY**: Term Type (e.g., `PN` for Preferred Name, `SY` for Synonym).
    - **SUPPRESS**: Whether this term type is suppressed by default (`N`, `O`, `Y`).
    """)
    return


@app.cell
def _(data_loader):
    mrrank_df = data_loader.load_ranking_metadata()
    return (mrrank_df,)


@app.cell
def _(mrrank_df):
    mrrank_df.sort_values(by='RANK', ascending=False).head(10)
    return


@app.cell
def _(mo, mrrank_df):
    # Example for MTH (Metathesaurus source)
    mth_ranking = mrrank_df[mrrank_df['SAB'] == 'MTH'].sort_values(by='RANK', ascending=False)

    mo.md(f"""
    ### 🔍 Ranking Insight: `MTH` (Metathesaurus)
    In the `MTH` source, the top-ranked term types are shown below. A rank of **{mth_ranking['RANK'].max() if not mth_ranking.empty else 'N/A'}** is assigned to the most preferred term type (**{mth_ranking['TTY'].iloc[0] if not mth_ranking.empty else 'N/A'}**).
    """)
    return (mth_ranking,)


@app.cell
def _(mth_ranking):
    mth_ranking.head(5)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 💡 What we can infer from this table:

    1.  **Top Priority**: The term type **PN** (Preferred Name) has the highest rank (**396**), meaning it is the first choice for a concept's name in `MTH`.
    2.  **Ranking Logic**: UMLS uses a numerical system where a **higher rank** indicates a higher preference.
    3.  **Active Status**: All top-ranked terms have `SUPPRESS = 'N'`, which means they are not suppressed and are active for use.
    4.  **Source Specificity**: These specific rankings only apply to the **MTH** source; other sources (like SNOMED CT) have their own ranking order.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Section 1 Summary: The Metadata Lookup Strategy

    To navigate the UMLS metadata efficiently, use this hierarchy when you encounter an unfamiliar term:

    | If you encounter a... | Look in... | Why? |
    | :--- | :--- | :--- |
    | **Column Header** (e.g., `TTY`, `SAB`, `CUI`) | [MRCOLS.RRF](#1.2-MRCOLS.RRF-Column-Definitions) | Defines the **Data Structure** and field meanings. |
    | **Code/Abbreviation** (e.g., `PT`, `isa`, `expanded_form`) | [MRDOC.RRF](#1.3-MRDOC.RRF-Metadata-Documentation) | Defines the **Internal Vocabulary** (key-value documentation). |
    | **Source Label** (e.g., `MSH`, `SNOMEDCT_US`) | [MRSAB.RRF](#1.4-MRSAB.RRF-Source-Vocabulary-Registry) | Defines the **Data Providers** (the "Who and When"). |
    | **Term Ranking** (e.g., `PN`, `SCD`, `0396`) | [MRRANK.RRF](#1.5-MRRANK.RRF-Concept-Name-Ranking) | Defines the **Preferred Term Ranking** within sources. |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Section 2: NET Directory (Semantic Network Metadata)

    In this section, we transition from the **Metathesaurus** (which contains concepts and their relationships) to the **Semantic Network**.

    The Semantic Network is the "Skeleton" or "Ontology" of UMLS. It provides a small, stable set of categories and relationships that provide a high-level abstraction of the medical domain.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.1 SRFIL (NET Directory Roadmap)

    **SRFIL** is the master catalog for the Semantic Network files. It provides a roadmap of the files included in the `NET` directory, their descriptions, and their physical attributes.

    ### Key Metrics:
    - **FIL**: File name.
    - **DES**: Brief description.
    - **FMT**: Column list (comma-separated).
    - **RWS**: Number of rows.
    """)
    return


@app.cell
def _(data_loader):
    srfil_df = data_loader.load_semantic_network_files()
    return (srfil_df,)


@app.cell
def _(srfil_df):
    srfil_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.2 SRFLD (Field descriptions for NET directory)

    **SRFLD** is the data dictionary for the Semantic Network. It defines every attribute (field) used across all files in the `NET` directory, providing descriptions and cross-references.

    ### Key Attributes:
    - **COL**: Column/Field name.
    - **DES**: Human-readable description.
    - **REF**: Reference to documentation.
    - **FIL**: The file(s) where this field appears.
    """)
    return


@app.cell
def _(data_loader):
    srfld_df = data_loader.load_semantic_network_fields()
    srfld_df.head(10)
    return (srfld_df,)


@app.cell
def _(mo, srfld_df):
    mo.md(f"""
    ### 📊 Semantic Network Metadata Summary
    - **Unique Fields**: {srfld_df['COL'].nunique()}
    - **Files Covered**: {srfld_df['FIL'].nunique()}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.3 SRDEF (Semantic Types & Relations Definitions)

    **SRDEF** serves as the primary registry for the Semantic Network. It defines the names, hierarchy, and properties of all **Semantic Types** (categories) and **Semantic Relations**.

    ### 2.3.1 Semantic Types
    These are the high-level categories used to group concepts in the Metathesaurus.

    ### Key Column Definitions:
    - **RT**: Record Type (`STY` for Semantic Type, `RL` for Relation).
    - **UI**: Unique Identifier (`T###` for types, `R###` for relations).
    - **NAME**: The official label (e.g., "Disease or Syndrome").
    - **TREE**: Hierarchy path (e.g., `A1.1.3` shows how types are nested).
    - **RIN**: The identifier of the Inverse Relation (only for `RL` records).
    """)
    return


@app.cell
def _(data_loader):
    srdef_df = data_loader.load_semantic_network_definitions()
    return (srdef_df,)


@app.cell
def _(srdef_df):
    # Split for easier analysis
    semantic_types_df = srdef_df[srdef_df['RT'] == 'STY'].copy()
    semantic_relations_df = srdef_df[srdef_df['RT'] == 'RL'].copy()
    return semantic_relations_df, semantic_types_df


@app.cell
def _(mo, semantic_relations_df, semantic_types_df):
    mo.md(f"""
    ### 📊 Inventory Summary
    - **Total Semantic Types (`STY`)**: {len(semantic_types_df)}
    - **Total Semantic Relations (`RL`)**: {len(semantic_relations_df)}

    #### Top 5 Semantic Types (Alphabetical)
    """)
    return


@app.cell
def _(semantic_types_df):
    semantic_types_df[['UI', 'NAME', 'TREE', 'DEF']].sort_values(by='NAME').head(5)
    return


@app.cell
def _(mo, semantic_types_df):
    # Analyze the hierarchy depth
    semantic_types_df['DEPTH'] = semantic_types_df['TREE'].str.count('\\.') + 1
    depth_stats = semantic_types_df['DEPTH'].value_counts().sort_index()

    mo.md(f"""
    ### 🌳 Hierarchy Analysis
    The semantic network has a maximum depth of **{semantic_types_df['DEPTH'].max()}** levels.

    #### Distribution of Semantic Types by Depth:
    """)
    return (depth_stats,)


@app.cell
def _(depth_stats):
    depth_stats
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📖 Guide: Reading the Hierarchy Path (`TREE`)
    The `TREE` field (also known as the Semantic Type Number or STN) uses a nested "breadcrumb" system. Each dot-separated segment represents a level of increasing specificity.

    **Example: `B2.2.1.2.1` (Disease or Syndrome)**

    | Level | Code | Name | Category Type |
    |:---|:---|:---|:---|
    | 1 | **B** | **Event** | Root Branch (Physical vs. Conceptual) |
    | 2 | **B2** | Phenomenon or Process | High-level Class |
    | 3 | **B2.2** | Natural Phenomenon or Process | Domain |
    | 4 | **B2.2.1** | Biologic Function | Context |
    | 5 | **B2.2.1.2**| Pathologic Function | Abnormal State (The Turning Point) |
    | 6 | **B2.2.1.2.1**| **Disease or Syndrome**| **Final Classification** |

    Tracing the path from left to right tells you the absolute lineage of any category in the UMLS ontology.
    """)
    return


@app.cell
def _(mo, semantic_types_df):
    # Deep dive into a common type: "Disease or Syndrome" (T047)
    _mask = semantic_types_df['UI'] == 'T047'
    _sample_type = semantic_types_df[_mask].iloc[0] if _mask.any() else semantic_types_df.iloc[0]

    mo.md(f"""
    ### 🔍 Semantic Type Deep Dive: `{_sample_type['NAME']}` ({_sample_type['UI']})
    - **Hierarchy Path (`TREE`)**: `{_sample_type['TREE']}`
    - **Definition**: *{_sample_type['DEF']}*
    - **Examples**: {_sample_type['EX'] or "None listed."}
    - **Abbreviation**: `{_sample_type['AB']}`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.3.2 Semantic Relations

    While Semantic Types categorize concepts, **Semantic Relations** define the potential linkages between those categories. For example, a "Biologic Function" might *affect* a "Disease or Syndrome".

    ### Key Features:
    - **Inverses**: Most relations have a counter-part (e.g., `part_of` vs. `has_part`).
    - **Symmetry**: Some relations are their own inverse (e.g., `temporally_related_to`).
    """)
    return


@app.cell
def _(mo, semantic_relations_df):
    # Analyze inverses
    relation_summary = semantic_relations_df[['UI', 'NAME', 'RIN', 'TREE', 'DEF']].sort_values(by='NAME')

    mo.md(f"""
    ### 🔗 Relation Registry & Inverse Mapping
    Below is the complete list of defined relations and their respective inverses:
    """)
    return (relation_summary,)


@app.cell
def _(relation_summary):
    relation_summary
    return


@app.cell
def _(mo, semantic_relations_df):
    # Identify symmetric vs asymmetric relations
    symmetric = semantic_relations_df[semantic_relations_df['NAME'] == semantic_relations_df['RIN']]
    asymmetric = semantic_relations_df[semantic_relations_df['NAME'] != semantic_relations_df['RIN']]

    mo.md(f"""
    ### ⚖️ Symmetry Analysis
    - **Symmetric Relations**: {len(symmetric)} (e.g., `{symmetric['NAME'].iloc[0] if not symmetric.empty else 'N/A'}`)
    - **Asymmetric Relations**: {len(asymmetric)} (e.g., `{asymmetric['NAME'].iloc[0] if not asymmetric.empty else 'N/A'}` → `{asymmetric['RIN'].iloc[0] if not asymmetric.empty else 'N/A'}`)
    """)
    return


@app.cell
def _(mo, relation_summary):
    # Identify top-level relations (depth 1)
    relation_summary['DEPTH'] = relation_summary['TREE'].str.count('\\.') + 1
    top_relations = relation_summary[relation_summary['DEPTH'] == 1]

    mo.md(f"""
    ### 🏘️ Relation Hierarchy (Root Branches)
    Just like Semantic Types, relations are organized into a hierarchy. There are **{len(top_relations)}** major branches of relationships:
    """)
    return (top_relations,)


@app.cell
def _(top_relations):
    top_relations[['UI', 'NAME', 'TREE', 'DEF']]
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Section 2 Summary: Navigating the Semantic Network

    The Semantic Network files provide the ontological framework for the Metathesaurus. Use this strategy to explore it:

    | If you want to... | Look in... | Description |
    | :--- | :--- | :--- |
    | **Identify a file** in the NET directory | [SRFIL](#2.1-SRFIL-NET-Directory-Roadmap) | The master catalog for Semantic Network files. |
    | **Understand a column/field** | [SRFLD](#2.2-SRFLD-Field-descriptions-for-NET-directory) | The data dictionary for Semantic Network attributes. |
    | **Lookup a Semantic Type** | [SRDEF](#2.3-SRDEF-Semantic-Types-&-Relations-Definitions) | Defines STY records, their hierarchy (`TREE`), and definitions. |
    | **Lookup a Semantic Relation** | [SRDEF](#2.3-SRDEF-Semantic-Types-&-Relations-Definitions) | Defines RL records and their inverse relationships. |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Section 3: Core Data Files

    In this section, we move into the **Core Data Files** of the Metathesaurus. These files contain the actual medical concepts, their names, relationships, and definitions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.1 MRDEF.RRF (Definitions)

    **MRDEF.RRF** contains semantic definitions for the concepts in the Metathesaurus. While not every concept has a definition, this file is crucial for adding context and meaning beyond simple term labels.

    ### Why Start Here?
    - **Scale**: At ~317,000 rows, it is relatively small compared to `MRCONSO` (~15M+ rows in recent releases).
    - **Purpose**: It provides the "What does this mean?" layer.
    - **Analogy**: After learning the names of concepts in `MRCONSO`, we use `MRDEF` to understand their actual definitions.

    ### Key Column Definitions:
    - **CUI**: Concept Unique Identifier.
    - **AUI**: Atom Unique Identifier (linking to a specific term in `MRCONSO`).
    - **SAB**: Source abbreviation (which vocabulary provided the definition).
    - **DEF**: The actual definition text.
    """)
    return


@app.cell
def _(data_loader):
    mrdef_df = data_loader.load_concept_definitions()
    return (mrdef_df,)


@app.cell
def _(mo, mrdef_df):
    total_definitions = len(mrdef_df)
    unique_cuis_with_def = mrdef_df['CUI'].nunique()

    mo.md(f"""
    ### 📊 Definitions Overview
    - **Total Definitions**: {total_definitions:,}
    - **Unique Concepts (CUIs) with Definitions**: {unique_cuis_with_def:,}

    #### Sample Definitions:
    """)
    return


@app.cell
def _(mrdef_df):
    mrdef_df.head(10)
    return


@app.cell
def _(mo, mrdef_df, mrsab_df):
    _counts = mrdef_df['SAB'].value_counts().reset_index()
    _counts.columns = ['RSAB', 'Definition Count']

    # Join with MRSAB to get full names and languages
    source_distribution = _counts.merge(
        mrsab_df[['RSAB', 'SON', 'LAT']].drop_duplicates('RSAB'),
        on='RSAB',
        how='left'
    )

    mo.md(f"""
    ### 📂 Source Distribution
    UMLS aggregates definitions from many sources. Below are the top 10 contributors of definitions, joined with their official metadata from `MRSAB.RRF`:
    """)
    return (source_distribution,)


@app.cell
def _(source_distribution):
    source_distribution.head(10)
    return


@app.cell
def _(mo, mrdef_df):
    # Let's pick a specific concept with a definition
    _sample_mask = mrdef_df['CUI'] == 'C0003123'
    _cui_definitions = mrdef_df[_sample_mask] if _sample_mask.any() else mrdef_df.head(1)
    sample_cui = _cui_definitions['CUI'].iloc[0]

    mo.md(f"""
    ### 🔍 Concept Deep Dive: `{sample_cui}`
    Let's look at the definitions available for a specific concept:
    """)
    return (sample_cui,)


@app.cell
def _(mrdef_df, sample_cui):
    # Show definitions for the same concept found above
    mrdef_df[mrdef_df['CUI'] == sample_cui]
    return


@app.cell
def _(mo, mrdef_df, sample_cui):
    _sample_row = mrdef_df[mrdef_df['CUI'] == sample_cui].iloc[0]

    mo.md(f"""
    ### 💡 Interpreting the Definition Record

    Using the record for `{sample_cui}` as an example:

    | Field | Value (from sample) | Meaning |
    | :--- | :--- | :--- |
    | **CUI** | `{_sample_row['CUI']}` | **Concept Unique Identifier**. The "Master ID" for this medical idea. |
    | **AUI** | `{_sample_row['AUI']}` | **Atom Unique Identifier**. Definitions are often tied to a specific term (atom) from the source. |
    | **SAB** | `{_sample_row['SAB']}` | **Source Abbreviation**. Tells us which vocabulary (e.g., `{_sample_row['SAB']}`) provided the definition. |
    | **DEF** | *"{_sample_row['DEF'][:150]}..."* | **The actual text**. This provides the semantic meaning for the concept. |
    | **SUPPRESS** | `{_sample_row['SUPPRESS']}` | **Suppress status**. `{_sample_row['SUPPRESS']}` means this is an active, non-suppressed definition. |

    **The Analogy:**
    If `MRCONSO` (which we'll explore next) is the dictionary index—telling you all the names for a concept—then `MRDEF` is the **dictionary entry itself**, providing the formal medical explanation of what it actually is.
    """)
    return


if __name__ == "__main__":
    app.run()
