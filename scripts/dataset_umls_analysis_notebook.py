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
    sample_row = mrfiles_df[_mask].iloc[0] if _mask.any() else mrfiles_df.iloc[0]

    mo.md(f"""
    ### 🔍 Row Interpretation: `{sample_row['FIL']}`
    Let's break down the entry for the Ambiguous LUI identifiers file:

    - **File Name (`FIL`)**: `{sample_row['FIL']}` (The physical file on disk)
    - **Description (`DES`)**: *{sample_row['DES']}*
    - **Columns (`FMT`)**: `{sample_row['FMT']}` (It contains {sample_row['CLS']} columns: {sample_row['FMT']})
    - **Scale**: It contains **{int(sample_row['RTY']):,}** rows and occupies **{int(sample_row['SZY']):,}** bytes.
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
    sample_col = mrcols_df[_mask].iloc[0] if _mask.any() else mrcols_df.iloc[0]

    mo.md(f"""
    ### 🔍 Column Interpretation: `{sample_col['COL']}` in `{sample_col['FIL']}`
    - **Description**: {sample_col['DES']}
    - **Data Type**: `{sample_col['DTY']}`
    - **Length Stats**: Min: {sample_col['MIN']}, Avg: {sample_col['AV']}, Max: {sample_col['MAX']}
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


if __name__ == "__main__":
    app.run()
