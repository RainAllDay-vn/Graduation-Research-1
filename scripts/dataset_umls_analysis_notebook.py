import marimo

__generated_with = "0.22.0"
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

    return KnowledgeGraph, UmlsDataLoader, mo, os, pd


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
    DATASET_ROOT = os.path.join("dataset", "umls", "extracted", "2025AB")
    _kg = KnowledgeGraph()
    data_loader = UmlsDataLoader(_kg.driver, dataset_path=DATASET_ROOT)
    return DATASET_ROOT, data_loader


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Section 1: Exploring `MRCONSO.RRF`

    `MRCONSO.RRF` is the **Concept Names and Sources** file. It contains the strings (names) associated with each CUI (Concept Unique Identifier) and tracks which source vocabularies they come from.
    """)
    return


@app.cell
def _(DATASET_ROOT, data_loader, mo, pd):
    headers = data_loader.load_headers_with_description(DATASET_ROOT, "MRCONSO.RRF")
    _header_df = pd.DataFrame(headers, columns=["Column", "Description"])

    _display_table = mo.vstack(
        [
            mo.md("### `MRCONSO.RRF` Column Definitions"),
            mo.ui.table(_header_df, pagination=True),
        ]
    )
    _display_table
    return


if __name__ == "__main__":
    app.run()
