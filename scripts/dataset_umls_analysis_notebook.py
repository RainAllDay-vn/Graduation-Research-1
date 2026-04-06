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
    DATASET_ROOT = os.path.join("dataset", "umls")
    _kg = KnowledgeGraph()
    data_loader = UmlsDataLoader(_kg.driver, dataset_path=DATASET_ROOT)
    return DATASET_ROOT, data_loader


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Section 1: Exploring `MRCONSO.RRF`

    `MRCONSO.RRF` is the **Concept Names and Sources** file. It contains the strings (names) associated with each CUI (Concept Unique Identifier) and tracks which source vocabularies they come from.

    ## Understanding the Columns in `MRCONSO.RRF`

    The **MRCONSO.RRF** file is the heart of the Unified Medical Language System (UMLS) Metathesaurus. It acts as a massive dictionary that maps medical terms from dozens of different vocabularies (like SNOMEDCT_US, ICD-10, and MeSH) to single, unified concepts.

    To understand these columns, it helps to first understand the UMLS hierarchy, which goes from broad to highly specific: **Concept (CUI)** $\rightarrow$ **Term (LUI)** $\rightarrow$ **String (SUI)** $\rightarrow$ **Atom (AUI)**.

    Here is a clearer, plain-English explanation of each column, grouped by their function:

    ### 1. The Core Hierarchy (From Broad to Specific)
    These IDs define how the UMLS organizes medical terms.

    * **CUI (Concept Unique Identifier):** Think of this as a single "folder" for a specific medical idea. No matter what language it's in or what source it comes from, if it means "Headache," it goes in the CUI folder for Headache (e.g., C0018681).
    * **LUI (Term Unique Identifier):** This groups together terms that are essentially the same word but perhaps pluralized or conjugated differently. For example, "Headache" and "Headaches" would share the same LUI.
    * **SUI (String Unique Identifier):** This is a unique ID for an *exact* string of text. If the spelling or capitalization changes at all, it gets a new SUI.
    * **AUI (Atom Unique Identifier):** The most granular level. An "Atom" is a specific string provided by a *specific source*. If ICD-10 provides the word "Headache" and SNOMED CT also provides the word "Headache", they will have the same SUI (exact text) and CUI (same meaning), but they will each get their own unique AUI because they came from different sources.

    ### 2. The Text Itself
    * **STR (String):** The actual text of the term you are looking at (e.g., "Myocardial Infarction", "Heart Attack", or "Infarto de miocardio").
    * **LAT (Language of Term):** The language the string is written in (e.g., `ENG` for English, `SPA` for Spanish).

    ### 3. Source Details (Where did this word come from?)
    These columns tell you which specific medical dictionary or terminology system provided this exact word.

    * **SAB (Source Abbreviation):** The abbreviation of the vocabulary that provided the term (e.g., `SNOMEDCT_US`, `ICD10CM`, `MSH` for MeSH).
    * **TTY (Term Type in Source):** What kind of term the source considers this to be. For example, is it their "Preferred Term" (`PT`), a "Synonym" (`SY`), or an "Abbreviation" (`AB`)?
    * **CODE (Source Code):** The official code that the *source vocabulary* uses for this term. For example, if the SAB is `ICD10CM` and the string is "Type 2 diabetes mellitus", the CODE would be `E11.9`.

    ### 4. Source-Asserted Identifiers (The Source's Own Internal IDs)
    Sometimes, the source vocabulary has its own complex internal database IDs. The UMLS stores them here so you can cross-reference them.

    * **SAUI (Source Asserted Atom Identifier):** The source's internal ID for this exact text string (Atom).
    * **SCUI (Source Asserted Concept Identifier):** The source's internal ID for this general concept.
    * **SDUI (Source Asserted Descriptor Identifier):** The source's internal ID for a broader category or descriptor this concept falls under.

    ### 5. Status and Preferences (Which term is the "best" one?)
    Since there are dozens of ways to say the same thing, UMLS tries to organize which terms are the primary ones to use.

    * **ISPREF (Is Preferred Atom?):** (`Y` or `N`). Indicates if this specific atom (from this specific source) is the preferred way to name the concept *within that source*.
    * **TS (Term Status):** (`P` or `S`). Indicates whether this term is the Preferred (`P`) or Secondary/Non-preferred (`S`) way to express the broader LUI (Term) group.
    * **STT (String Type):** Indicates if this exact string is the preferred spelling (`PF` - Preferred Form) or a variant (`VW` - Variant Word, `VC` - Variant Case) for its broader term group.

    ### 6. Administrative Flags
    These are used by developers to filter the database.

    * **SRL (Source Restriction Level):** A number (usually 0, 1, 2, 3, or 4) indicating the copyright or licensing restrictions of the source. `0` means free to use, while higher numbers mean you may need a special license from the vocabulary creator to use it in commercial software.
    * **SUPPRESS (Suppressible Flag):** (`O`, `E`, `Y`, or `N`). Tells you if this term should be hidden from normal searches. For example, it might be an Obsolete (`O`) term no longer used in medicine, or a term with an Error (`E`).
    * **CVF (Content View Flag):** A numeric bitmask. The UMLS creates pre-packaged "subsets" of the database for specific uses (like a subset just for clinical natural language processing). This flag indicates which pre-packaged subsets this row belongs to.
    """)
    return


@app.cell
def _(data_loader, mo, pd):
    headers = data_loader.load_headers_with_description("MRCONSO.RRF")
    _header_df = pd.DataFrame(headers, columns=["Column", "Description"])
    _sample = data_loader.load_file("META", "MRCONSO.RRF", offset=0, limit=1)
    _sample = _sample.iloc[0]
    _sample = _sample.values
    _header_df['Example Value'] = _sample
    
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
