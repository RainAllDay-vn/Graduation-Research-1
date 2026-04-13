import marimo

__generated_with = "0.23.1"
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

    return KnowledgeGraph, UmlsDataLoader, mo, os, plt


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
    def _get_data_loader():
        DATASET_ROOT = os.path.join("dataset", "umls")
        kg = KnowledgeGraph()
        return UmlsDataLoader(kg.driver, dataset_path=DATASET_ROOT)

    data_loader = _get_data_loader()
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
    mrfiles_df.head(10)
    return (mrfiles_df,)


@app.cell
def _(mo, mrfiles_df):
    def _get_row_interpretation():
        mask = mrfiles_df['FIL'] == 'AMBIGLUI.RRF'
        sample_row = mrfiles_df[mask].iloc[0] if mask.any() else mrfiles_df.iloc[0]

        return mo.md(f"""
        ### 🔍 Row Interpretation: `{sample_row['FIL']}`
        Let's break down the entry for the Ambiguous LUI identifiers file:

        - **File Name (`FIL`)**: `{sample_row['FIL']}` (The physical file on disk)
        - **Description (`DES`)**: *{sample_row['DES']}*
        - **Columns (`FMT`)**: `{sample_row['FMT']}` (It contains {sample_row['CLS']} columns: {sample_row['FMT']})
        - **Scale**: It contains **{int(sample_row['RTY']):,}** rows and occupies **{int(sample_row['SZY']):,}** bytes.
        """)

    _get_row_interpretation()
    return


@app.cell
def _(mo, mrfiles_df):
    def _get_dataset_summary():
        total_files = len(mrfiles_df)
        total_records = mrfiles_df['RTY'].sum()

        return mo.md(f"""
        ### 📊 Dataset Summary
        - **Total Files**: {total_files}
        - **Total Records (All Files)**: {int(total_records):,}

        #### Top 10 Largest Files (by Record Count)
        """)

    _get_dataset_summary()
    return


@app.cell
def _(mrfiles_df):
    def _get_top_10_largest_files():
        _df = mrfiles_df.sort_values(by='RTY', ascending=False)
        _df = _df[['FIL', 'DES', 'RTY']]
        _df = _df.reset_index(drop=True)
        return _df.head(10)

    _get_top_10_largest_files()
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
    mrcols_df.head(10)
    return (mrcols_df,)


@app.cell
def _(mo, mrcols_df):
    def _get_column_interpretation():
        mask = mrcols_df['COL'] == 'CUI'
        sample_col = mrcols_df[mask].iloc[0] if mask.any() else mrcols_df.iloc[0]

        return mo.md(f"""
        ### 🔍 Column Interpretation: `{sample_col['COL']}` in `{sample_col['FIL']}`
        - **Description**: {sample_col['DES']}
        - **Data Type**: `{sample_col['DTY']}`
        - **Length Stats**: Min: {sample_col['MIN']}, Avg: {sample_col['AV']}, Max: {sample_col['MAX']}
        """)

    _get_column_interpretation()
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
    mrdoc_df[mrdoc_df['TYPE'] == 'rel_inverse'].head(10)
    return (mrdoc_df,)


@app.cell
def _(mo, mrcols_df, mrdoc_df):
    def _get_dataset_overview():
        def get_info(key):
            desc = mrcols_df[mrcols_df['COL'] == key]['DES'].iloc[0] if key in mrcols_df['COL'].values else "Metadata Category"
            return desc

        dockey_counts = mrdoc_df['DOCKEY'].value_counts().to_frame(name='count')
        dockey_counts['description'] = dockey_counts.index.map(get_info)

        header = mo.md(f"""
        ### 📊 Dataset Overview
        The documentation catalog contains **{len(dockey_counts)}** distinct categories. Below is the master view of the documentation map and the most frequent documentation types (**DOCKEY**):
        """)

        atn_view = mrdoc_df[mrdoc_df['DOCKEY'] == 'ATN']
        rela_view = mrdoc_df[mrdoc_df['DOCKEY'] == 'RELA']

        view = mo.vstack([
            mo.md("#### Frequencies"),
            dockey_counts,
            mo.md("#### Previews"),
            mo.md("**Attributes (ATN):**"),
            atn_view.head(5),
            mo.md("**Relationship Labels (RELA):**"),
            rela_view.head(5)
        ])

        return mo.vstack([header, view])

    _get_dataset_overview()
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
    mrsab_df.head(10)
    return (mrsab_df,)


@app.cell
def _(mo, mrsab_df):
    def _get_global_reach():
        lang_counts = mrsab_df['LAT'].value_counts()

        md = mo.md(f"""
        ### 🌍 Global Reach
        The current release includes contents in **{len(lang_counts)}** languages.

        #### Sources by Language (Top 10)
        """)

        return mo.vstack([
            md,
            lang_counts
        ])

    _get_global_reach()
    return


@app.cell
def _(mo, mrsab_df):
    def _get_source_detailed_preview():
        major_source = mrsab_df[mrsab_df['RSAB'] == 'MSH'].iloc[0] if 'MSH' in mrsab_df['RSAB'].values else mrsab_df.iloc[0]

        return mo.md(f"""
        ### 🔍 Source Detailed Preview: `{major_source['RSAB']}`
        - **Official Name**: {major_source['SON']}
        - **Family**: {major_source['SF'] or 'N/A'}
        - **Version**: `{major_source['SVER']}`
        - **Language**: {major_source['LAT']}
        - **Scale**: Contains **{int(major_source['CFR']):,}** concepts (CUIs).
        """)

    _get_source_detailed_preview()
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
    mrrank_df.sort_values(by='RANK', ascending=False).head(10)
    return (mrrank_df,)


@app.cell
def _(mo, mrrank_df):
    def _get_ranking_insight():
        mth_ranking = mrrank_df[mrrank_df['SAB'] == 'MTH'].sort_values(by='RANK', ascending=False)

        return mo.md(f"""
        ### 🔍 Ranking Insight: `MTH` (Metathesaurus)
        In the `MTH` source, the top-ranked term types are shown below. A rank of **{mth_ranking['RANK'].max() if not mth_ranking.empty else 'N/A'}** is assigned to the most preferred term type (**{mth_ranking['TTY'].iloc[0] if not mth_ranking.empty else 'N/A'}**).
        """)

    _get_ranking_insight()
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
    srfil_df.head(10)
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
def _(data_loader, mo):
    srfld_df = data_loader.load_semantic_network_fields()

    def _get_semantic_network_summary():
        return mo.md(f"""
        ### 📊 Semantic Network Metadata Summary
        - **Unique Fields**: {srfld_df['COL'].nunique()}
        - **Files Covered**: {srfld_df['FIL'].nunique()}
        """)

    _get_semantic_network_summary()
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
    - **STY/RL**: The official label (e.g., "Disease or Syndrome").
    - **STN/RTN**: Hierarchy path (e.g., `A1.1.3` shows how types are nested).
    - **DEF**: Definition text.
    - **EX**: Examples of concepts in this category.
    - **ABR**: Abbreviation.
    - **RIN**: The identifier of the Inverse Relation (only for `RL` records).
    """)
    return


@app.cell
def _(data_loader):
    srdef_df = data_loader.load_semantic_network_definitions()
    return (srdef_df,)


@app.cell
def _(mo, srdef_df):
    def _split_srdef():
        semantic_types_df = srdef_df[srdef_df['RT'] == 'STY']
        return mo.vstack([
            mo.md(f"**Semantic Types** ({len(semantic_types_df)} entries - Showing 5):"),
            semantic_types_df.head(5)
        ])

    _split_srdef()
    return


@app.cell
def _(mo, srdef_df):
    def _get_hierarchy_analysis():
        semantic_types_df = srdef_df[srdef_df['RT'] == 'STY']
        # Use STN/RTN instead of TREE
        semantic_types_df['DEPTH'] = semantic_types_df['STN/RTN'].str.count('\\.') + 1
        depth_stats = semantic_types_df['DEPTH'].value_counts().sort_index()

        md = mo.md(f"""
        ### 🌳 Hierarchy Analysis
        The semantic network has a maximum depth of **{semantic_types_df['DEPTH'].max()}** levels.

        #### Distribution of Semantic Types by Depth:
        """)

        return mo.vstack([
            md,
            depth_stats
        ])

    _get_hierarchy_analysis()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📖 Guide: Reading the Hierarchy Path (`STN/RTN`)
    The `STN/RTN` field (also known as the Semantic Type Number or STN) uses a nested "breadcrumb" system. Each dot-separated segment represents a level of increasing specificity.

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
def _(mo, srdef_df):
    def _get_semantic_type_deep_dive():
        semantic_types_df = srdef_df[srdef_df['RT'] == 'STY']
        mask = semantic_types_df['UI'] == 'T047'
        sample_type = semantic_types_df[mask].iloc[0] if mask.any() else semantic_types_df.iloc[0]

        return mo.md(f"""
        ### Sample entry: `{sample_type['STY/RL']}` ({sample_type['UI']})
        - **Hierarchy Path (`STN/RTN`)**: `{sample_type['STN/RTN']}`
        - **Definition**: *{sample_type['DEF']}*
        - **Examples**: {sample_type['EX'] or "None listed."}
        - **Abbreviation**: `{sample_type['ABR']}`
        """)

    _get_semantic_type_deep_dive()
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
def _(mo, srdef_df):
    def _split_srdef():
        semantic_relations_df = srdef_df[srdef_df['RT'] == 'RL']
        return mo.vstack([
            mo.md(f"**Semantic Relations** ({len(semantic_relations_df)} entries - Showing 5):"),
            semantic_relations_df.head(5)
        ])

    _split_srdef()
    return


@app.cell
def _(mo, srdef_df):
    def _get_relation_registry():
        semantic_relations_df = srdef_df[srdef_df['RT'] == 'RL']
        # Use STY/RL instead of NAME and STN/RTN instead of TREE
        relation_summary = semantic_relations_df[['UI', 'STY/RL', 'RIN', 'DEF']].sort_values(by='STY/RL').reset_index(drop=True)

        text = mo.md(f"""
        ### 🔗 Relation Registry & Inverse Mapping
        Below is some of the defined relations and their respective inverses:
        """)

        return mo.vstack([
            text,
            relation_summary.head(10)
        ])

    _get_relation_registry()
    return


@app.cell
def _(mo, srdef_df):
    def _get_symmetry_analysis():
        semantic_relations_df = srdef_df[srdef_df['RT'] == 'RL']
        # Use STY/RL instead of NAME
        symmetric = semantic_relations_df[semantic_relations_df['STY/RL'] == semantic_relations_df['RIN']]
        asymmetric = semantic_relations_df[semantic_relations_df['STY/RL'] != semantic_relations_df['RIN']]

        return mo.md(f"""
        ### ⚖️ Symmetry Analysis
        - **Symmetric Relations**: {len(symmetric)} (e.g., `{symmetric['STY/RL'].iloc[0] if not symmetric.empty else 'N/A'}`)
        - **Asymmetric Relations**: {len(asymmetric)} (e.g., `{asymmetric['STY/RL'].iloc[0] if not asymmetric.empty else 'N/A'}` → `{asymmetric['RIN'].iloc[0] if not asymmetric.empty else 'N/A'}`)
        """)

    _get_symmetry_analysis()
    return


@app.cell
def _(mo, srdef_df):
    def _get_relation_hierarchy():
        # Filter for Relations (RL)
        semantic_relations_df = srdef_df[srdef_df['RT'] == 'RL'].copy()

        # Calculate depth based on tree dots (e.g., 'R1.1.2' has 2 dots, depth 3)
        # Use STN/RTN instead of TREE
        semantic_relations_df['DEPTH'] = semantic_relations_df['STN/RTN'].str.count('\\.') + 1

        # Summarize counts per depth level
        depth_counts = semantic_relations_df['DEPTH'].value_counts().sort_index().reset_index()
        depth_counts.columns = ['Depth Level', 'Number of Relations']

        md = mo.md(f"""
        ### 🏘️ Relation Hierarchy (Root Branches)
        Just like Semantic Types, relations are organized into a hierarchy. There are **{len(depth_counts)}** distinct levels of depth in the relationship tree:
        """)

        return mo.vstack([
            md,
            mo.ui.table(depth_counts) # Assuming marimo's table UI for better readability
        ])

    _get_relation_hierarchy()
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
    ## 3.1 MRCONSO.RRF (Concept Names and Sources)

    **MRCONSO.RRF** is the most important file in the UMLS Metathesaurus. It is the central mapping table that links every concept (CUI) to its various names, codes from different vocabularies, and internal identifiers.

    ### Key Concepts:
    - **CUI**: Concept Unique Identifier (The "Master ID").
    - **AUI**: Atom Unique Identifier (The specific "Name" from a source).
    - **SAB**: Source Abbreviation (Who provided this name).
    - **STR**: String (The actual name text).

    ### Hierarchy of Term Identifiers:
    1. **CUI** (Concept): The core semantic idea.
    2. **LUI** (Lexical): Terms with the same normalized form.
    3. **SUI** (String): Specific text variations (plural, case, etc).
    4. **AUI** (Atom): The unique occurrence from a specific source.
    """)
    return


@app.cell
def _(data_loader, mo):
    mrconso_df = data_loader.load_concepts(limit=1_000_000)
    def _get_mrconso_scale_analysis():
        total_rows = len(mrconso_df)
        unique_cuis = mrconso_df['CUI'].nunique()
        unique_auis = mrconso_df['AUI'].nunique()

        # Pick a sample CUI (Macroaggregated Albumin or the first concept)
        sample_mask = mrconso_df['CUI'] == 'C0000005'
        cui_terms = mrconso_df[sample_mask] if sample_mask.any() else mrconso_df.head(1)


        return mo.md(f"""
        ### 📊 MRCONSO Scale & Quality Analysis
        - **Total Records (Atoms)**: {total_rows:,}
        - **Unique Concepts (CUIs)**: {unique_cuis:,}
        - **Unique Atoms (AUIs)**: {unique_auis:,}
        - **Synonymy Ratio**: {total_rows/unique_cuis:.2f} names per concept on average.
        """)

    _get_mrconso_scale_analysis()
    return (mrconso_df,)


@app.cell
def _():
    conso_sample_cui = 'C0003123'
    return (conso_sample_cui,)


@app.cell
def _(mo, mrconso_df, plt):
    def _plot_dist(col, title):
        # 1. Calculate normalized counts (percentages)
        counts = mrconso_df[col].value_counts(normalize=True)

        # 2. Group everything under 5% into 'Others'
        mask = counts >= 0.03
        plot_data = counts[mask]

        if not counts[~mask].empty:
            others_val = counts[~mask].sum()
            plot_data['Others'] = others_val

        # 3. Create the Matplotlib figure
        # We use subplots to get a figure object for marimo to display
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.pie(
            plot_data, 
            labels=plot_data.index, 
            autopct='%1.1f%%', 
            startangle=140,
            colors=plt.cm.Paired.colors
        )
        ax.set_title(title)

        # 4. Clean up and return
        plt.close(fig)  # Prevents duplicate rendering in some environments
        return mo.vstack([
            mo.md(f"#### {title}"),
            fig
        ])

    mo.vstack([
        _plot_dist('LAT', 'Top Languages'),
        _plot_dist('SAB', 'Top Sources'),
        _plot_dist('TTY', 'Top Term Types'),
        _plot_dist('SUPPRESS', 'Suppress Status')
    ], justify='start')
    return


@app.cell
def _(conso_sample_cui, mo, mrconso_df):
    def _get_concept_interpretation():
        sample_row = mrconso_df[mrconso_df['CUI'] == conso_sample_cui].iloc[0]

        return mo.md(f"""
        ### 💡 Interpreting the Concept Atom (MRCONSO)

        `MRCONSO` contains 18 columns that together define the lineage and metadata of every term. Using the record for `{conso_sample_cui}` as an example:

        | Field | Value | Meaning |
        | :--- | :--- | :--- |
        | **CUI** | `{sample_row['CUI']}` | **Concept Unique Identifier**. The global ID shared by all synonyms. |
        | **LAT** | `{sample_row['LAT']}` | **Language**. The language of this specific name (e.g., `{sample_row['LAT']}`). |
        | **TS** | `{sample_row['TS']}` | **Term Status**. `P` for Preferred, `S` for Synonym. |
        | **LUI** | `{sample_row['LUI']}` | **Lexical Unique Identifier**. Groups terms with the same normalized form. |
        | **STT** | `{sample_row['STT']}` | **String Type**. `PF` (Preferred form), `VC` (Variant), etc. |
        | **SUI** | `{sample_row['SUI']}` | **String Unique Identifier**. Unique ID for this exact text string. |
        | **ISPREF** | `{sample_row['ISPREF']}` | **Is Preferred**. `Y/N` if this atom is preferred in its source. |
        | **AUI** | `{sample_row['AUI']}` | **Atom Unique Identifier**. The unique occurrence of this term in this source. |
        | **SAUI** | `{sample_row['SAUI'] or 'N/A'}` | **Source Atom ID**. The original ID from the provider (if available). |
        | **SCUI** | `{sample_row['SCUI'] or 'N/A'}` | **Source Concept ID**. The provider's concept identifier. |
        | **SDUI** | `{sample_row['SDUI'] or 'N/A'}` | **Source Descriptor ID**. The provider's descriptor identifier. |
        | **SAB** | `{sample_row['SAB']}` | **Source Abbreviation**. Which vocabulary provided this name. |
        | **TTY** | `{sample_row['TTY']}` | **Term Type**. The role of this name (e.g., `PT` for primary term). |
        | **CODE** | `{sample_row['CODE']}` | **Source Code**. The specific code or ID in the source vocabulary. |
        | **STR** | *"{sample_row['STR']}"* | **String**. The actual name or label of the concept. |
        | **SRL** | `{sample_row['SRL']}` | **Source Restriction Level**. Used for licensing/copyright logic. |
        | **SUPPRESS** | `{sample_row['SUPPRESS']}` | **Suppress status**. `N` (active), `O` (obsolete), `Y` (suppressed). |
        | **CVF** | `{sample_row['CVF'] or 'N/A'}` | **Content View Flag**. Used to identify special subsets of data. |

        **The Analogy:**
        If UMLS is a social network, the **CUI** is the **User Account (Identity)**. The **LUI/SUI/AUI** hierarchy tracks how that same person might use different spellings, plural forms, or handles (Atoms) across different platforms (Sources).
        """)

    _get_concept_interpretation()
    return


@app.cell
def _(mo, mrconso_df):
    def _get_preferred_str_analysis():
        total_cuis = mrconso_df['CUI'].nunique()
        preferred_counts = mrconso_df[mrconso_df['ISPREF'] == 'Y'].groupby('CUI')['STR'].nunique()

        no_preferred = total_cuis - len(preferred_counts)
        exactly_one = (preferred_counts == 1).sum()
        more_than_one = (preferred_counts > 1).sum()

        # Filter for English Preferred terms
        eng_pref_df = mrconso_df[(mrconso_df['ISPREF'] == 'Y') & (mrconso_df['LAT'] == 'ENG')]

        # CUIs with at least one preferred term in English
        eng_any_pref = eng_pref_df['CUI'].nunique()

        # CUIs with exactly one preferred term in English
        eng_preferred_counts = eng_pref_df.groupby('CUI')['STR'].nunique()
        eng_exactly_one = (eng_preferred_counts == 1).sum()

        return mo.md(f"""
        ### 🎯 Preferred STR Analysis
        - **Total CUIs**: {total_cuis:,}
        - **CUIs with NO preferred term**: {no_preferred:,} ({no_preferred/total_cuis*100:.1f}%)
        - **CUIs with exactly ONE preferred STR**: {exactly_one:,} ({exactly_one/total_cuis*100:.1f}%)
        - **CUIs with MORE than one preferred STR**: {more_than_one:,} ({more_than_one/total_cuis*100:.1f}%)
        - **CUIs with ANY preferred term in English**: {eng_any_pref:,} ({eng_any_pref/total_cuis*100:.1f}%)
        - **CUIs with exactly ONE preferred STR in English**: {eng_exactly_one:,} ({eng_exactly_one/total_cuis*100:.1f}%)
        """)

    _get_preferred_str_analysis()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.2 MRDEF.RRF (Definitions)

    **MRDEF.RRF** contains semantic definitions for the concepts in the Metathesaurus. While not every concept has a definition, this file is crucial for adding context and meaning beyond simple term labels.

    ### Why Explore this?
    - **Scale**: At ~317,000 rows, it is much smaller than `MRCONSO`.
    - **Purpose**: It provides the "What does this mean?" layer.
    - **Analogy**: After learning the names of concepts in **MRCONSO (3.1)**, we use **MRDEF** to understand their actual definitions.

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
    def _get_definitions_overview():
        total_definitions = len(mrdef_df)
        unique_cuis_with_def = mrdef_df['CUI'].nunique()

        md = mo.md(f"""
        ### 📊 Definitions Overview
        - **Total Definitions**: {total_definitions:,}
        - **Unique Concepts (CUIs) with Definitions**: {unique_cuis_with_def:,}

        #### Sample Definitions:
        """)

        return mo.vstack([
            md,
            mrdef_df.head(10)
        ])    

    _get_definitions_overview()
    return


@app.cell
def _(mo, mrdef_df, mrsab_df):
    def _get_source_distribution():
        counts = mrdef_df['SAB'].value_counts().reset_index()
        counts.columns = ['RSAB', 'Definition Count']

        # Join with MRSAB to get full names and languages
        source_distribution = counts.merge(
            mrsab_df[['RSAB', 'SON', 'LAT']].drop_duplicates('RSAB'),
            on='RSAB',
            how='left'
        )

        md = mo.md(f"""
        ### 📂 Source Distribution
        UMLS aggregates definitions from many sources. Below are the top 10 contributors of definitions, joined with their official metadata from `MRSAB.RRF`:
        """)

        return mo.vstack([
            md,
            source_distribution
        ])

    _get_source_distribution()
    return


@app.cell
def _(conso_sample_cui, mo, mrdef_df):
    def _get_concept_deep_dive():
        # Filter for the specific concept
        sample_row = mrdef_df[mrdef_df['CUI'] == conso_sample_cui]

        # Handle case where no row is found to prevent index errors
        if sample_row.empty:
            return mo.md(f"No definition found for CUI: `{conso_sample_cui}`")

        # Extract values safely using .iloc[0]
        cui_val = sample_row['CUI'].iloc[0]
        aui_val = sample_row['AUI'].iloc[0]
        sab_val = sample_row['SAB'].iloc[0]
        def_val = sample_row['DEF'].iloc[0]
        sup_val = sample_row['SUPPRESS'].iloc[0]

        return mo.vstack([
            mo.md(f"""
            ### 🔍 Concept Deep Dive: `{conso_sample_cui}`
            Let's look at the definitions available for a specific concept:
            """),
            sample_row,
            mo.md(f"""
            #### Interpretation

            | Field | Value (from sample) | Meaning |
            | :--- | :--- | :--- |
            | **CUI** | `{cui_val}` | **Concept Unique Identifier**. The "Master ID" for this medical idea. |
            | **AUI** | `{aui_val}` | **Atom Unique Identifier**. Definitions are often tied to a specific term (atom) from the source. |
            | **SAB** | `{sab_val}` | **Source Abbreviation**. Tells us which vocabulary (e.g., `{sab_val}`) provided the definition. |
            | **DEF** | *"{def_val[:150]}..."* | **The actual text**. This provides the semantic meaning for the concept. |
            | **SUPPRESS** | `{sup_val}` | **Suppress status**. `N` usually means this is an active, non-suppressed definition. |

            **The Analogy:**
            If `MRCONSO` is the dictionary index—telling you all the names for a concept—then `MRDEF` is the **dictionary entry itself**, providing the formal medical explanation of what it actually is.
            """)
        ])

    _get_concept_deep_dive()
    return


@app.cell
def _(conso_sample_cui, mo, mrconso_df, mrdef_df):
    def _get_random_sample_comparison():
        term_name = mrconso_df[mrconso_df['CUI'] == conso_sample_cui]['STR'].iloc[0]
        defn_series = mrdef_df[mrdef_df['CUI'] == conso_sample_cui]['DEF']

        definition_elements = [
            mo.md(f"**{i}.** {text}") 
            for i, text in enumerate(defn_series, 1)
        ]

        return mo.vstack([
            mo.md('''
            #### Comparing Names to Definitions

            Let's explore how much more detail the definitions provide compared to just the term names (STR) from MRCONSO.
            '''),
            mo.md(f"**Name (STR)**: {term_name}"),
            mo.md(f"**Definitions (DEF)**:"),
            *definition_elements
        ])

    _get_random_sample_comparison()
    return


@app.cell
def _(mo, mrconso_df, mrdef_df):
    def _get_cui_distribution():
        total_rows = len(mrdef_df)
        unique_cuis = mrdef_df['CUI'].nunique()
        duplicate_cuis = mrdef_df['CUI'].value_counts()
        most_common_cui = duplicate_cuis.idxmax()
        most_common_count = duplicate_cuis.max()

        md1 = mo.md(f"""
        ### 📊 CUI Distribution Analysis
        In MRDEF, the number of unique CUIs is less than the total number of rows. This is because a single concept can have multiple definitions from different sources.

        - **Total Rows**: {total_rows:,}
        - **Unique CUIs**: {unique_cuis:,}
        - **Difference**: {total_rows - unique_cuis:,} rows have duplicate CUIs

        This means on average each CUI has **{total_rows/unique_cuis:.2f}** definitions from different sources.
        """)

        md2 = mo.md(f"""
        CUI with Most Definitions is `{most_common_cui}`. This concept has **{most_common_count}** different definitions from various sources.
        """)

        df = mrdef_df[mrdef_df['CUI'] == most_common_cui]
        df = df.sample(n=10, random_state = 42).reset_index(drop=True)
        df = df[['CUI', 'DEF']]

        total_unique_cuis_conso = mrconso_df['CUI'].nunique()
        unique_cuis_with_def = mrdef_df['CUI'].nunique()
        percentage = (unique_cuis_with_def / total_unique_cuis_conso) * 100

        md3 = mo.md(f'''
        ### 📈 Definition Coverage
        Let's calculate what percentage of concepts in MRCONSO have at least one definition in MRDEF.
        - **Total Unique CUIs in MRCONSO**: {total_unique_cuis_conso:,}
        - **CUIs with Definitions in MRDEF**: {unique_cuis_with_def:,}
        - **Coverage**: **{percentage:.2f}%**

        This means that less than 10% of the concepts in UMLS have formal definitions available. Many concepts are represented only by their names/synonyms without detailed definitions.
        ''')

        return mo.vstack([
            md1,
            md2,
            df,
            md3
        ])

    _get_cui_distribution()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.3 MRSTY.RRF (Semantic Types)

    **MRSTY.RRF** maps every concept in the Metathesaurus to one or more **Semantic Types** from the Semantic Network. While MRCONSO tells us "what" concepts are called and MRDEF provides definitions, MRSTY tells us "what kind" of thing they are.

    ### Why Explore this?
    - **Scale**: With ~3.1 million rows, it's a substantial mapping file.
    - **Purpose**: It provides the ontological categorization that enables semantic reasoning, filtering, and inference.
    - **Analogy**: If MRCONSO is the dictionary and MRDEF is the definitions, then MRSTY is the **category system** that groups all terms into high-level domains like "Disease or Syndrome", "Pharmacologic Substance", "Therapeutic Procedure", etc.

    ### Key Column Definitions:
    - **CUI**: Concept Unique Identifier.
    - **TUI**: Semantic Type Unique Identifier (e.g., T047).
    - **STN**: Semantic Type Number (hierarchical path from SRDEF, e.g., B2.2.1.2.1).
    - **STY**: Semantic Type label (e.g., "Disease or Syndrome").
    - **ATUI**: Atom Unique Identifier (links to a specific term in MRCONSO).
    - **CVF**: Content View Flag.
    """)
    return


@app.cell
def _(data_loader):
    mrsty_df = data_loader.load_semantic_types()
    return (mrsty_df,)


@app.cell
def _(mo, mrsty_df):
    def _get_scale_analysis():
        total_rows = len(mrsty_df)
        unique_cuis = mrsty_df['CUI'].nunique()
        unique_tuis = mrsty_df['TUI'].nunique()
        unique_stys = mrsty_df['STY'].nunique()

        md = mo.md(f"""
        ### 📊 MRSTY Scale Analysis
        - **Total Records**: {total_rows:,}
        - **Unique Concepts (CUIs)**: {unique_cuis:,}
        - **Unique Semantic Types (STYs)**: {unique_stys:,}
        - **Unique TUIs**: {unique_tuis:,}
        - **Average Semantic Types per Concept**: {total_rows/unique_cuis:.2f}

        #### Sample values:
        """)

        return mo.vstack([
            md,
            mrsty_df.head(10)
        ])

    _get_scale_analysis()
    return


@app.cell
def _(conso_sample_cui, mo, mrsty_df, srdef_df):
    def _get_semantic_type_deep_dive():
        # Get the TUI for the sample CUI
        sample_sty_row = mrsty_df[mrsty_df['CUI'] == conso_sample_cui].iloc[0]
        sample_tui = sample_sty_row['TUI']

        # UI in SRDEF matches TUI in MRSTY
        type_info = srdef_df[srdef_df['UI'] == sample_tui].iloc[0]
        concept_count = (mrsty_df['TUI'] == sample_tui).sum()

        return mo.md(f"""
        #### Example entry: `{type_info['STY/RL']}` ({sample_tui})

        - **Hierarchy Path (`STN/RTN`)**: `{type_info['STN/RTN']}`
        - **Definition**: *{type_info['DEF']}*
        - **Examples**: {type_info['EX'] or "None listed."}
        - **Abbreviation**: `{type_info['ABR']}`
        - **Number of Concepts in Metathesaurus**: {concept_count:,}
        """)

    _get_semantic_type_deep_dive()
    return


@app.cell
def _(mo, mrsty_df):
    def _get_semantic_type_distribution():
        sty_counts = mrsty_df['STY'].value_counts().reset_index()
        sty_counts.columns = ['Semantic Type', 'Concept Count']

        md = mo.md(f"""
        ### 📊 Semantic Type Distribution (Top 10)
        The table below shows the 10 most common semantic types in the UMLS Metathesaurus:
        """)

        return mo.vstack([
            md,
            sty_counts.head(10)
        ])

    _get_semantic_type_distribution()
    return


@app.cell
def _(mo, mrsty_df):
    def _get_multiple_types_analysis():
        type_counts = mrsty_df.groupby('CUI')['STY'].nunique()
        single_type = (type_counts == 1).sum()
        multiple_types = (type_counts > 1).sum()

        # Find concepts with the most semantic types
        max_types = type_counts.max()
        multi_type_cuis = type_counts[type_counts == max_types].index.tolist()
        sample_cui = multi_type_cuis[0] if multi_type_cuis else None

        md = mo.md(f"""
        ### 🎯 Multiple Semantic Types Analysis
        - **Concepts with ONE semantic type**: {single_type:,} ({single_type/len(type_counts)*100:.1f}%)
        - **Concepts with MULTIPLE semantic types**: {multiple_types:,} ({multiple_types/len(type_counts)*100:.1f}%)
        - **Maximum semantic types per concept**: {max_types}
        """)

        if sample_cui:
            sample_types = mrsty_df[mrsty_df['CUI'] == sample_cui][['STY', 'TUI']]
            md2 = mo.md(f"""
            ### 📋 Sample Concept with Multiple Types: `{sample_cui}`
            This concept has **{max_types}** semantic types:
            """)
            return mo.vstack([md, md2, sample_types])

        return md

    _get_multiple_types_analysis()
    return


@app.cell
def _(mo, mrsty_df, srdef_df):
    def _get_hierarchy_coverage():
        semantic_types_df = srdef_df[srdef_df['RT'] == 'STY']
        total_types_in_srdef = len(semantic_types_df)

        # Get unique TUIs from MRSTY
        unique_tuis_in_mrsty = mrsty_df['TUI'].nunique()

        # Find types defined in SRDEF but not used in MRSTY
        unused_types = set(semantic_types_df['UI']) - set(mrsty_df['TUI'])

        return mo.md(f"""
        ### 🏛️ Hierarchy Coverage Analysis

        The Semantic Network defines a comprehensive taxonomy of **{total_types_in_srdef}** semantic types. We have that:

        - **Total Semantic Types Defined**: {total_types_in_srdef}
        - **Types Used in MRSTY**: {unique_tuis_in_mrsty}
        - **Unused Types**: {len(unused_types)}

        This indicates that **{unique_tuis_in_mrsty/total_types_in_srdef*100:.1f}%** of all defined semantic types are actively used to categorize concepts in the current UMLS release.
        """)

    _get_hierarchy_coverage()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.4 MRSAT.RRF (Attributes)

    **MRSAT.RRF** contains additional attributes for concepts, terms, and strings. This file is the "catch-all" for metadata that is too specific or diverse to fit into the fixed columns of other files like `MRCONSO` or `MRSTY`.

    ### Key Column Definitions:
    - **CUI**: Concept Unique Identifier.
    - **STYPE**: The type of object the attribute is attached to (e.g., `CUI`, `AUI`, `CODE`).
    - **ATN**: Attribute Name (e.g., `DEFINITION`, `SEMANTIC_TYPE`).
    - **ATV**: Attribute Value (the actual data).
    - **SAB**: Source Abbreviation.
    """)
    return


@app.cell
def _(data_loader):
    # MRSAT is very large (~50M rows), so we load a representative sample for analysis
    mrsat_df = data_loader.load_attributes(limit=1_000_000)
    return (mrsat_df,)


@app.cell
def _(mo, mrsat_df):
    def _get_mrsat_scale_analysis():
        _total_rows = len(mrsat_df)
        _unique_cuis = mrsat_df['CUI'].nunique()
        _unique_atns = mrsat_df['ATN'].nunique()

        return mo.md(f"""
        ### 📊 MRSAT Sample Analysis (First 100k rows)
        - **Total Records in Sample**: {_total_rows:,}
        - **Unique Concepts (CUIs)**: {_unique_cuis:,}
        - **Unique Attribute Names (ATN)**: {_unique_atns:,}

        #### Sample Rows:
        """)

    _get_mrsat_scale_analysis()
    return


@app.cell
def _(mrsat_df):
    mrsat_df.head(10)
    return


@app.cell
def _(mo, mrdoc_df, mrsat_df):
    def _get_attachment_level_analysis():
        _stype_counts = mrsat_df['STYPE'].value_counts().reset_index()
        _stype_counts.columns = ['STYPE', 'Count']

        # Join with MRDOC to get expanded names
        _stype_labels = mrdoc_df[mrdoc_df['DOCKEY'] == 'STYPE'][['VALUE', 'EXPL']].rename(columns={'VALUE': 'STYPE', 'EXPL': 'Description'})
        _stype_analysis = _stype_counts.merge(_stype_labels, on='STYPE', how='left')
        _stype_analysis = _stype_analysis.rename(columns={'STYPE': 'Attachment Level (STYPE)'})

        return mo.vstack([
            mo.md(f"""
            ### 🎯 Attachment Level Analysis (`STYPE`)
            Attributes in `MRSAT` can be attached to different levels of the UMLS hierarchy:
            """),
            _stype_analysis
        ])

    _get_attachment_level_analysis()
    return


@app.cell
def _(mo, mrdoc_df, mrsat_df):
    def _get_attribute_name_distribution():
        _atn_counts = mrsat_df['ATN'].value_counts().head(15).reset_index()
        _atn_counts.columns = ['ATN', 'Count']

        # Join with MRDOC to get expanded names
        _atn_labels = mrdoc_df[mrdoc_df['DOCKEY'] == 'ATN'][['VALUE', 'EXPL']].rename(columns={'VALUE': 'ATN', 'EXPL': 'Description'})
        _atn_analysis = _atn_counts.merge(_atn_labels, on='ATN', how='left')

        return mo.vstack([
            mo.md(f"""
            ### 📈 Top 15 Attribute Names (`ATN`)
            Attributes often represent source-specific identifiers, flags, or specialized medical metadata.
            """),
            _atn_analysis
        ])

    _get_attribute_name_distribution()
    return


@app.cell
def _(conso_sample_cui, mo, mrsat_df):
    def _get_concept_attributes_deep_dive():
        _sample_attrs = mrsat_df[mrsat_df['CUI'] == conso_sample_cui]

        if _sample_attrs.empty:
            return mo.md(f"No attributes found for CUI `{conso_sample_cui}` in the current 100k sample.")

        return mo.vstack([
            mo.md(f"### 🔍 Concept Attributes Deep Dive: `{conso_sample_cui}`"),
            _sample_attrs[['STYPE', 'ATN', 'ATV', 'SAB']]
        ])

    _get_concept_attributes_deep_dive()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.5 MRHIER.RRF (Hierarchies)

    **MRHIER.RRF** contains computable hierarchies for concepts in the Metathesaurus. It explicitly represents the parent-child relationships and provides the full path to the root for each concept within a specific source.

    ### Key Column Definitions:
    - **CUI**: Concept Unique Identifier.
    - **AUI**: Atom Unique Identifier.
    - **CXN**: Context Number (distinguishes multiple locations in a hierarchy).
    - **PAUI**: Parent Atom Unique Identifier.
    - **SAB**: Source Abbreviation.
    - **RELA**: Relationship Label (e.g., `isa`, `part_of`).
    - **PTR**: Path to Root (a dot-separated list of AUIs from the root down to the parent).
    - **HCD**: Hierarchy Code.
    """)
    return


@app.cell
def _(data_loader):
    # MRHIER is large (~21M rows), loading a sample
    mrhier_df = data_loader.load_hierarchies(limit=1_000_000)
    return (mrhier_df,)


@app.cell
def _(mo, mrhier_df):
    def _get_mrhier_scale_analysis():
        _total_rows = len(mrhier_df)
        _unique_cuis = mrhier_df['CUI'].nunique()
        _unique_sabs = mrhier_df['SAB'].nunique()

        return mo.md(f"""
        ### 📊 MRHIER Sample Analysis (First 1M rows)
        - **Total Records in Sample**: {_total_rows:,}
        - **Unique Concepts (CUIs)**: {_unique_cuis:,}
        - **Unique Sources (SAB)**: {_unique_sabs:,}

        #### Sample Rows:
        """)

    _get_mrhier_scale_analysis()
    return


@app.cell
def _(mrhier_df):
    mrhier_df.head(10)
    return


@app.cell
def _(conso_sample_cui, mo, mrhier_df):
    def _get_hierarchy_interpretation():
        # Try to find the sample CUI, otherwise take the first row
        _mask = mrhier_df['CUI'] == conso_sample_cui
        _sample_row = mrhier_df[_mask].iloc[0] if _mask.any() else mrhier_df.iloc[0]

        return mo.md(f"""
        ### 🔍 Hierarchy Entry Interpretation: `{_sample_row['CUI']}`
        - **Source (`SAB`)**: `{_sample_row['SAB']}`
        - **Parent Atom (`PAUI`)**: `{_sample_row['PAUI'] or 'ROOT'}`
        - **Relationship (`RELA`)**: `{_sample_row['RELA'] or 'N/A'}`
        - **Path to Root (`PTR`)**: `{_sample_row['PTR'] or 'N/A'}`

        **What this tells us:**
        The `PTR` field is particularly powerful for graph traversal as it encodes the entire lineage of the concept within the source hierarchy without needing recursive joins.
        """)

    _get_hierarchy_interpretation()
    return


@app.cell
def _(mo, mrhier_df):
    def _get_depth_analysis():
        # Calculate depth based on the number of dots in the PTR (Path To Root)
        # PTR is AUI1.AUI2.AUI3...
        _depths = mrhier_df['PTR'].str.split('.').str.len().fillna(0)
        _depth_counts = _depths.value_counts().sort_index().reset_index()
        _depth_counts.columns = ['Hierarchy Depth', 'Count']

        return mo.vstack([
            mo.md("### 🌳 Hierarchy Depth Analysis"),
            mo.md("Distribution of concepts by their depth in the source hierarchy:"),
            _depth_counts
        ])

    _get_depth_analysis()
    return


@app.cell
def _(mo, mrhier_df):
    def _get_rela_distribution():
        _rela_counts = mrhier_df['RELA'].value_counts().reset_index()
        _rela_counts.columns = ['Relationship Label (RELA)', 'Count']

        return mo.vstack([
            mo.md("### 🔗 Relationship Types in Hierarchies"),
            _rela_counts
        ])

    _get_rela_distribution()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.6 MRREL.RRF (Relationships)

    **MRREL.RRF** contains relationship records between concepts, atoms, or other identifiers. It is the core file for understanding how different concepts in the Metathesaurus are connected to each other, beyond simple hierarchies.

    ### Key Column Definitions:
    - **CUI1**: The first Concept Unique Identifier in the relationship.
    - **AUI1**: The first Atom Unique Identifier.
    - **REL**: Broad relationship type (e.g., `PAR` for parent, `CHD` for child, `RO` for other).
    - **RELA**: Relationship attribute (more specific relationship, e.g., `isa`, `part_of`).
    - **CUI2**: The second Concept Unique Identifier.
    - **AUI2**: The second Atom Unique Identifier.
    - **SAB**: Source Abbreviation.
    - **STYPE1/STYPE2**: The type of identifier for CUI1/AUI1 and CUI2/AUI2 (e.g., `CUI`, `AUI`).
    """)
    return


@app.cell
def _(data_loader):
    # MRREL is very large, loading a sample
    mrrel_df = data_loader.load_relationships(limit=1_000_000)
    return (mrrel_df,)


@app.cell
def _(mo, mrrel_df):
    def _get_mrrel_scale_analysis():
        _total_rows = len(mrrel_df)
        _unique_cuis_1 = mrrel_df['CUI1'].nunique()
        _unique_cuis_2 = mrrel_df['CUI2'].nunique()
        _unique_rels = mrrel_df['REL'].nunique()
        _unique_relas = mrrel_df['RELA'].nunique()

        return mo.md(f"""
        ### 📊 MRREL Sample Analysis (First 1M rows)
        - **Total Records in Sample**: {_total_rows:,}
        - **Unique Concepts in CUI1**: {_unique_cuis_1:,}
        - **Unique Concepts in CUI2**: {_unique_cuis_2:,}
        - **Broad Relationship Types (REL)**: {_unique_rels}
        - **Specific Relationship Attributes (RELA)**: {_unique_relas}

        #### Sample Rows:
        """)

    _get_mrrel_scale_analysis()
    return


@app.cell
def _(mrrel_df):
    mrrel_df.head(10)
    return


@app.cell
def _(conso_sample_cui, mo, mrrel_df):
    def _get_relationship_deep_dive():
        # Try to find relationships for the sample CUI
        _mask = (mrrel_df['CUI1'] == conso_sample_cui) | (mrrel_df['CUI2'] == conso_sample_cui)
        _sample_rels = mrrel_df[_mask].reset_index(drop=True)

        if _sample_rels.empty:
            return mo.md(f"No relationships found for CUI `{conso_sample_cui}` in the current 1M sample.")

        return mo.vstack([
            mo.md(f"### 🔍 Concept Relationships Deep Dive: `{conso_sample_cui}`"),
            _sample_rels[['CUI1', 'REL', 'RELA', 'CUI2', 'SAB']].head(10)
        ])

    _get_relationship_deep_dive()
    return


@app.cell
def _(mo, mrdoc_df, mrrel_df):
    def _get_rel_distribution():
        _rel_counts = mrrel_df['REL'].value_counts().reset_index()

        # Join with MRDOC to get expanded names
        _rel_labels = mrdoc_df[mrdoc_df['TYPE'] == 'expanded_form'][['VALUE', 'EXPL']].rename(columns={'VALUE': 'REL', 'EXPL': 'Description'})
        _rel_analysis = _rel_counts.merge(_rel_labels, on='REL', how='left')
        _rel_analysis = _rel_analysis.rename(columns={'REL': 'Broad Relation (REL)'})

        return mo.vstack([
            mo.md("### 📊 Broad Relationship Distribution (REL)"),
            _rel_analysis
        ])

    _get_rel_distribution()
    return


@app.cell
def _(mo, mrdoc_df, mrrel_df):
    def _get_rela_distribution():
        _rela_counts = mrrel_df['RELA'].value_counts().head(15).reset_index()
        _rela_counts.columns = ['RELA', 'Count']

        # Join with MRDOC to get expanded names
        _rela_labels = mrdoc_df[mrdoc_df['DOCKEY'] == 'RELA'][['VALUE', 'EXPL', 'TYPE']].rename(columns={'VALUE': 'RELA', 'EXPL': 'Description'})
        _rela_analysis = _rela_counts.merge(_rela_labels, on='RELA', how='left')
        _rela_analysis = _rela_analysis.rename(columns={'RELA': 'Specific Relation (RELA)'})

        return mo.vstack([
            mo.md("### 📊 Top 15 Relationship Attributes (RELA)"),
            _rela_analysis
        ])

    _get_rela_distribution()
    return


@app.cell
def _(mrrel_df):
    mrrel_df[mrrel_df['RELA'] == ''].head()
    return


@app.cell
def _(mo, mrdoc_df, mrrel_df):
    def _investigate_empty_rela():
        # Filter for rows where RELA is empty or whitespace
        _empty_mask = (mrrel_df['RELA'] == '') | (mrrel_df['RELA'].isna())
        _empty_rela_df = mrrel_df[_empty_mask]

        # Group by REL to see broad relationship types
        _rel_counts = _empty_rela_df['REL'].value_counts().reset_index()
        _rel_counts.columns = ['REL', 'Count']

        # Join with MRDOC to get descriptions
        _rel_labels = mrdoc_df[mrdoc_df['DOCKEY'] == 'REL'][['VALUE', 'EXPL']].rename(columns={'VALUE': 'REL', 'EXPL': 'Description'})
        _rel_analysis = _rel_counts.merge(_rel_labels, on='REL', how='left')
        _rel_analysis = _rel_analysis.rename(columns={'REL': 'Broad Relation (REL)'})

        return mo.vstack([
            mo.md(f"### 🔍 Investigation: Empty RELA Rows"),
            mo.md(f"Total rows with empty `RELA`: **{len(_empty_rela_df):,}**"),
            mo.md("#### Breakdown by Broad Relation (REL):"),
            _rel_analysis
        ])

    _investigate_empty_rela()
    return


@app.cell
def _(mo, mrrel_df):
    def _check_bidirectionality():
        # Create a set of unique (CUI1, CUI2) pairs
        _pairs_set = set(zip(mrrel_df['CUI1'], mrrel_df['CUI2']))

        # Calculate how many pairs have an inverse in the same sample
        _two_way = 0
        _total = len(_pairs_set)

        for _c1, _c2 in _pairs_set:
            if (_c2, _c1) in _pairs_set:
                _two_way += 1

        return mo.md(f"""
        ### 🔄 Bidirectionality Check (Symmetry)

        This check investigates if concept pairs `(CUI1, CUI2)` have a corresponding reverse relationship `(CUI2, CUI1)` within the current sample.

        - **Total Unique Concept Pairs (C1, C2)**: {_total:,}
        - **Bidirectional Pairs**: {_two_way:,} ({_two_way/_total*100:.2f}%)
        - **Unidirectional Pairs (in this sample)**: {_total - _two_way:,} ({(_total - _two_way)/_total*100:.2f}%)

        > [!WARNING]
        > UMLS relationships are designed to be two-way (e.g., if A is a parent of B, then B is a child of A). However, since we are using a **sample of 1,000,000 rows**, the inverse record for many relationships might be located outside of this sample, leading to a higher "unidirectional" count than reality.
        """)

    _check_bidirectionality()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Section 3 Summary: Core Data Files Overview

    The core data files we've explored provide the foundation of the UMLS Metathesaurus:

    | File | Primary Purpose | Key Insight |
    | :--- | :--- | :--- |
    | **MRCONSO.RRF** (3.1) | Concept Names & Sources | Maps every concept to all its names/synonyms across vocabularies. |
    | **MRDEF.RRF** (3.2) | Definitions | Provides formal semantic definitions for concepts (where available). |
    | **MRSTY.RRF** (3.3) | Semantic Types | Categorizes each concept into high-level ontological categories. |
    | **MRSAT.RRF** (3.4) | Attributes | Catch-all for specialized source-specific metadata (flags, IDs, tags). |
    | **MRHIER.RRF** (3.5) | Hierarchies | Explicit parent-child paths and full lineage (PTR). |
    | **MRREL.RRF** (3.6) | Relationships | Connections between concepts, atoms, and other identifiers. |

    ### The Semantic Hierarchy
    These six files work together to create a rich semantic layer:

    1. **Identity**: MRCONSO tells us "what is this concept called?"
    2. **Meaning**: MRDEF tells us "what does this concept mean?"
    3. **Category**: MRSTY tells us "what kind of thing is this?"
    4. **Extensibility**: MRSAT tells us "what else do we know about this from specific sources?"
    5. **Lineage**: MRHIER tells us "where does this concept fit in the tree?"
    6. **Connection**: MRREL tells us "how is this concept related to other concepts?"

    This hexad enables sophisticated semantic reasoning, query expansion, and cross-vocabulary integration in medical informatics applications.
    """)
    return


if __name__ == "__main__":
    app.run()
