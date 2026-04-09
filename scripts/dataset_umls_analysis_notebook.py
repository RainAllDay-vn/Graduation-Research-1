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

    return KnowledgeGraph, UmlsDataLoader, mo, os, pd, plt


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


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1.1 The Core Hierarchy (From Broad to Specific)

    This group has 4 different types of ID which help differentiate each concepts as well as their alternative names or description
    """)
    return


@app.cell
def _(data_loader, pd):
    _total_rows = 1_000_000
    sample = data_loader.load_file("META", "MRCONSO.RRF", offset=0, limit=_total_rows)

    results = pd.DataFrame(
        {
            "Identifier": ["CUI (Concepts)", "LUI (Terms)", "SUI (Strings)", "AUI (Atoms)", "Total Rows"],
            "Count": [
                len(sample["CUI"].unique()),
                len(sample["LUI"].unique()),
                len(sample["SUI"].unique()),
                len(sample["AUI"].unique()),
                _total_rows,
            ],
        }
    )
    results["Count"] = results["Count"].apply(lambda x: f"{x:,}")
    results
    return (sample,)


@app.cell
def _(mo, sample):
    # 1. Find CUI with largest number of AUIs
    top_cui = sample.groupby("CUI")["AUI"].count().sort_values(ascending=False).index[0]
    top_cui_data = sample[sample["CUI"] == top_cui]

    _stats = {
        "CUI": top_cui,
        "Unique LUIs": len(top_cui_data["LUI"].unique()),
        "Unique SUIs": len(top_cui_data["SUI"].unique()),
        "Total AUIs": len(top_cui_data["AUI"]),
        "Representative Name": top_cui_data["STR"].iloc[0],
    }

    _display = mo.vstack(
        [
            mo.md(f"### Analysis of CUI: `{top_cui}`"),
            mo.md(f"We choose this concept as an example because this concept has the highest number of rows in the sample. Representative name: **{_stats['Representative Name']}**."),
            mo.ui.table([_stats]),
        ]
    )
    _display
    return (top_cui_data,)


@app.cell
def _(mo, top_cui_data):
    top_lui = top_cui_data["LUI"].value_counts().idxmax()
    _top_lui_count = top_cui_data["LUI"].value_counts().max()
    top_lui_data = top_cui_data[top_cui_data["LUI"] == top_lui]
    _different_lui_data = top_cui_data.groupby("LUI").first()[["AUI", "STR"]]
    _same_lui_data = top_lui_data.groupby("SUI").first()[["AUI", "STR"]]

    _display = mo.vstack(
        [
            mo.md(f"This concept has {len(top_cui_data["LUI"].unique())} unique `LUI` values. Entries with different `LUI` show different terms for the same concepts. Meanwhile, entries with the same `LUI` but different `SUI` show different capitalization or spelling for the same term."),
            mo.md("For example, here are the entries for the same concept `CUI` but with different `LUI`:"),
            mo.ui.table(_different_lui_data),
            mo.md(f"Now, consider `{top_lui}` which has {_top_lui_count}, we have the follwing entries:"),
            mo.ui.table(_same_lui_data),
            mo.md(f"They are all different spelling or capitalization of {_same_lui_data["STR"].iloc[0]}")
        ]
    )
    _display
    return (top_lui_data,)


@app.cell
def _(mo, top_lui_data):
    top_sui = top_lui_data["SUI"].value_counts().idxmax()
    _top_sui_count = top_lui_data["SUI"].value_counts().max()
    top_sui_data  = top_lui_data[top_lui_data["SUI"] == top_sui][["SUI", "SAB", "TTY", "STR"]]


    _display = mo.vstack(
        [
            mo.md(f"Lastly, consider `SUI`. With the same `SUI` value, we can view different source for the same `STR` value. For example, here are the entries {top_sui} with {_top_sui_count} entries:"),
            mo.ui.table(top_sui_data)
        ]
    )
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1.2: Language Distribution Analysis

    This section analyzes the distribution of different languages in the sample set. The `LAT` column indicates the language of each term.
    """)
    return


@app.cell
def _(mo, pd, plt, sample):
    _counts = sample["LAT"].value_counts().reset_index()
    _counts.columns = ["Language", "Count"]

    # Calculate percentage
    _total = _counts["Count"].sum()
    _counts["Percentage"] = (_counts["Count"] / _total) * 100

    # Group languages with less than 1% to "Other"
    _mask = _counts["Percentage"] < 1
    _other_count = _counts[_mask]["Count"].sum()
    _main_groups = _counts[~_mask].copy()

    if _other_count > 0:
        _other_row = pd.DataFrame({"Language": ["Other"], "Count": [_other_count], "Percentage": [(_other_count / _total) * 100]})
        _main_groups = pd.concat([_main_groups, _other_row], ignore_index=True)

    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.pie(_main_groups["Count"], labels=_main_groups["Language"], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    _ax.set_title("Language Distribution (Minority Groups < 1% merged as 'Other')")

    _display = mo.vstack(
        [
            mo.md("### Language Distribution Pie Chart"),
            mo.as_html(_fig),
            mo.md("### Detailed Language Counts"),
            mo.ui.table(_main_groups[["Language", "Count", "Percentage"]]),
        ]
    )
    plt.close(_fig)
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Observations:
    *   **English Dominance**: English (`ENG`) is overwhelmingly the most common language in the sample, accounting for approximately **94.5%** of the entries.
    *   **German (GER) Representation**: German is the second most frequent language at **5.1%**.
    *   **Minority Languages**: Combined, all other languages account for less than **0.4%**.
    *   **Metathesaurus Focus**: This distribution highlights the UMLS Metathesaurus's primary grounding in English terminology, which is critical when training or evaluating query generation models.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1.3: Source Distribution Analysis

    This section analyzes the distribution of different source vocabularies in the sample set. The `SAB` (Source Abbreviation) column identifies the origin of each term.
    """)
    return


@app.cell
def _(mo, pd, plt, sample):
    _counts = sample["SAB"].value_counts().reset_index()
    _counts.columns = ["Source", "Count"]

    # Calculate percentage
    _total = _counts["Count"].sum()
    _counts["Percentage"] = (_counts["Count"] / _total) * 100

    # Group sources with less than 2% to "Other"
    _mask = _counts["Percentage"] < 2
    _other_count = _counts[_mask]["Count"].sum()
    _main_groups = _counts[~_mask].copy()

    if _other_count > 0:
        _other_row = pd.DataFrame({"Source": ["Other"], "Count": [_other_count], "Percentage": [(_other_count / _total) * 100]})
        _main_groups = pd.concat([_main_groups, _other_row], ignore_index=True)

    _fig, _ax = plt.subplots(figsize=(10, 10))
    _ax.pie(_main_groups["Count"], labels=_main_groups["Source"], autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
    _ax.set_title("Source Distribution (Minority Groups < 2% merged as 'Other')")

    _display = mo.vstack(
        [
            mo.md("### Source Distribution Pie Chart"),
            mo.as_html(_fig),
            mo.md("### Detailed Source Counts"),
            mo.ui.table(_main_groups[["Source", "Count", "Percentage"]]),
        ]
    )
    plt.close(_fig)
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Observations:
    *   **Dominant Sources**: **SNOMEDCT_US** (30.3%) and **MSH** (27.4%) together account for more than 57% of all entries in the sample, highlighting their central role in the UMLS Metathesaurus.
    *   **Secondary Sources**: **CHV** (7.2%), **MSHGER** (5.1%), and **NCI** (4.5%) provide significant additional terminology.
    *   **Diversity**: Smaller sources like **ICD9CM** (2.8%) and **LOINC/LNC** (2.0%) are also present, despite the 2% grouping threshold.
    *   **Long Tail**: The "Other" category accounts for 15.0% of the sample, indicating a broad range of specialized vocabularies that each contribute less than 2% individually.
    """)
    return


if __name__ == "__main__":
    app.run()
