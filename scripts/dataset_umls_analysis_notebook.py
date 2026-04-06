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
    from scripts.models import DataLoader
    from scripts.knowledge_graph import KnowledgeGraph
    from scripts.dataset_specific.umls.data_loader import get_loader

    return Any, DataLoader, Dict, Driver, List, Optional, mo, np, os, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Section 0: Setting up

    This notebook performs a comprehensive analysis of the **UMLS (Unified Medical Language System)** dataset.
    The goal is to understand the structure, scale, and content of the Metathesaurus files before processing or integrating them into the Knowledge Graph.
    """)
    return


@app.cell
def _(os):
    # Base configuration
    DATASET_ROOT = os.path.join("dataset", "umls", "extracted", "2025AB")
    kg = KnowledgeGraph()
    data_loader = get_loader(kg.driver, dataset_path=DATASET_ROOT)
    return (DATASET_ROOT,)

if __name__ == "__main__":
    app.run()
