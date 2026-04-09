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


if __name__ == "__main__":
    app.run()
