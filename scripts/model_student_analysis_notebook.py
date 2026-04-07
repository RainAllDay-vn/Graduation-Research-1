import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import sqlite3
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import re
    import logging
    import ast
    from typing import Dict, Any, Tuple, List
    from CyVer import SyntaxValidator
    from neo4j import GraphDatabase, basic_auth
    from model_evaluator import ModelEvaluator

    # Silence verbose Neo4j notifications
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    return (
        Any,
        Dict,
        GraphDatabase,
        List,
        ModelEvaluator,
        SyntaxValidator,
        Tuple,
        ast,
        basic_auth,
        mo,
        pd,
        plt,
        re,
    )


@app.cell
def _(mo):
    mo.md("""
    # Model Analysis: Reasoning Performance

    This notebook analyzes the performance of **Qwen/Qwen3.5-0.8B** on the **LC-QuAD 2.0** dataset when reasoning is enabled.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data Loading and Preparation

    We retrieve cached model responses from a SQLite database (`ai_cache.db`).
    For **Reasoning ON**, we parse the content to separate the `<think>` reasoning block from the final Cypher query.
    """)
    return


@app.cell
def _(ModelEvaluator, pd):
    evaluator = ModelEvaluator(model_name='dummy', api_key='')

    # Load reasoning=1 data with all fields
    df_reason = evaluator.fetch_cached_responses('Qwen/Qwen3.5-0.8B', 'lc-quad-2.0', include_reasoning=1)
    df_reason.rename(columns={'content': 'full_content'}, inplace=True)

    def _extract_parts(text):
        if not text:
            return None
        _clean_text = text.replace("<think>", "")
        before, sep, after = _clean_text.partition("</think>")
        before, after = before.strip(), after.strip()

        if not before or not sep or not after:
            return None
        return before, after

    _parts = df_reason['full_content'].apply(_extract_parts)
    invalid_response_count = int(_parts.isna().sum())
    _valid_parts = _parts.dropna()

    # Initialize columns with empty/None
    df_reason['thinking_text'] = ""
    df_reason['query_text'] = ""

    # Map valid parts back to the dataframe
    if not _valid_parts.empty:
        df_reason = df_reason.loc[_valid_parts.index].copy()
        df_reason['thinking_text'] = _valid_parts.apply(lambda x: x[0])
        df_reason['query_text'] = _valid_parts.apply(lambda x: x[1])

    df_reason['thinking_length'] = pd.to_numeric(df_reason['thinking_text'].str.len(), errors='coerce')
    df_reason['query_length'] = pd.to_numeric(df_reason['query_text'].str.len(), errors='coerce')
    return df_reason, invalid_response_count


@app.cell
def _(mo):
    mo.md("""
    ## Data Preview
    """)
    return


@app.cell
def _(df_reason, mo):
    _preview = mo.vstack([
        mo.md("### Preview: Reasoning ON"),
        mo.ui.table(df_reason.head())
    ])
    _preview
    return


@app.cell
def _(df_reason, invalid_response_count, mo):
    if df_reason.empty:
        _stats_md = mo.md("# Missing data for analysis.")
    else:
        _s_re = df_reason['query_length'].describe()
        _s_th = df_reason['thinking_length'].describe()

        # Create a small warning string if there are invalid responses
        _invalid_msg = ""
        if invalid_response_count > 0:
            _invalid_msg = f"**Note:** {invalid_response_count} responses were skipped because they were missing `</think>` tags."

        _stats_md = mo.md(f"""
        ## Summary Statistics: Reasoning Performance

        This table analyzes the character lengths:
        - **ON: Query**: The extracted query section.
        - **ON: Thinking**: The reasoning content extracted from `<think>` tags.

        | Metric | ON: Query | ON: Thinking |
        | :--- | :--- | :--- |
        | **Count** | {_s_re['count']:.0f} | {_s_th['count']:.0f} |
        | **Mean** | {_s_re['mean']:.2f} | {_s_th['mean']:.2f} |
        | **Median (50%)** | {_s_re['50%']:.2f} | {_s_th['50%']:.2f} |
        | **Std Dev** | {_s_re['std']:.2f} | {_s_th['std']:.2f} |
        | **Min** | {_s_re['min']:.0f} | {_s_th['min']:.0f} |
        | **Max** | {_s_re['max']:.0f} | {_s_th['max']:.0f} |

        {_invalid_msg}
        """)

    _stats_md
    return


@app.cell
def _(df_reason, mo, pd):
    def _get_samples(df, text_col, n=5):
        if df.empty:
            return pd.DataFrame()

        # Ensure variety by dropping identical queries first
        _unique_df = df.drop_duplicates(subset=[text_col])
        _n = min(n, len(_unique_df))

        # Random sample for a diverse view
        return _unique_df.sample(n=_n, random_state=42)

    _samples_on = _get_samples(df_reason, 'query_text')

    def _format_list(samples, text_col):
        if samples.empty:
            return "_No samples available._"

        items = []
        for _, _row in samples.iterrows():
            _question = _row['question'] if 'question' in _row else "Unknown Question"
            _length = int(_row['query_length'])
            _text = _row[text_col].strip() if _row[text_col] else "[Empty]"

            # Format each sample with its question and the Cypher query
            _item_md = (
                f"**Question:** {_question}\n\n"
                f"**Query (Length: {_length} chars):**\n"
                f"```cypher\n{_text}\n```"
            )
            items.append(_item_md)

        return "\n\n---\n\n".join(items)

    _content = mo.md(f"""
    ## Random Sample Queries

    These samples are selected at random from the dataset to provide a diverse look at the model's output quality.

    ### Reasoning ON (Average: {df_reason['query_length'].mean() if not df_reason.empty else 0:.1f} chars)
    {_format_list(_samples_on, 'query_text')}
    """)

    _content
    return


@app.cell
def _(mo):
    mo.md("""
    ## Distribution Analysis

    The following visualizations analyze the character length distributions of the generated Cypher queries when reasoning is enabled.
    Note that we only measure the *final query* length, excluding the thinking process.
    """)
    return


@app.cell
def _(df_reason, plt):
    _fig = None
    if not df_reason.empty:
        _l1, _u1 = df_reason['query_length'].quantile(0.05), df_reason['query_length'].quantile(0.95)

        _fig, _ax = plt.subplots(figsize=(10, 6))

        _ax.hist(df_reason['query_length'], bins=50, range=(_l1, _u1), 
                 alpha=0.6, label='Reasoning ON (Extracted Query)', color='orange', edgecolor='darkorange')

        _ax.set_title("Query Length Distribution: Reasoning ON")
        _ax.set_xlabel("Query Length (characters)")
        _ax.set_ylabel("Frequency")
        _ax.legend()
        _ax.grid(True, alpha=0.3)
        plt.tight_layout()

    _fig
    return


@app.cell
def _(df_reason, plt):
    _fig = None
    if not df_reason.empty:
        _u1 = df_reason['query_length'].quantile(0.95)

        _fig, _ax = plt.subplots(figsize=(10, 4))
        _ax.boxplot(df_reason['query_length'], vert=False, labels=['Reasoning ON (Query)'], 
                    patch_artist=True, boxprops=dict(facecolor='orange', alpha=0.6))

        _ax.set_xlim(0, _u1) 

        _ax.set_title("Response Length Box Plot (Reasoning ON)")
        _ax.set_xlabel("Length (characters)")
        _ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Thinking Process Analysis

    When reasoning is enabled, the model generates a "thinking" chain before producing the final query.
    Below is the distribution of the length (in characters) of these reasoning paths.
    """)
    return


@app.cell
def _(df_reason, plt):
    _fig = None
    if not df_reason.empty:
        _l, _u = df_reason['thinking_length'].quantile(0.05), df_reason['thinking_length'].quantile(0.95)
        _df_mid = df_reason[(df_reason['thinking_length'] >= _l) & (df_reason['thinking_length'] <= _u)]

        _fig, (_ax_hist, _ax_box) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), 
                                                 gridspec_kw={'height_ratios': [3, 1]},
                                                 sharex=True)

        _ax_hist.hist(_df_mid['thinking_length'], bins=50, color='purple', edgecolor='black', alpha=0.7)
        _ax_hist.set_title("Distribution of Thinking Length (Reasoning ON)")
        _ax_hist.set_ylabel("Frequency")
        _ax_hist.grid(axis='y', alpha=0.3)

        _ax_box.boxplot(df_reason['thinking_length'], vert=False, patch_artist=True, 
                        boxprops=dict(facecolor='plum'))
        _ax_box.set_xlabel("Thinking Length (characters)")
        _ax_box.set_yticks([])
        _ax_box.grid(axis='x', alpha=0.3)

        plt.tight_layout()

    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cypher Query Syntax Validation (CyVer AST)

    We validate all generated Cypher queries using **CyVer**'s `SyntaxValidator`, which uses the openCypher ANTLR grammar under the hood to perform real AST-based syntax validation.

    This analysis determines what percentage of the model's generated queries are syntactically invalid.
    """)
    return


@app.cell
def _(Any, Dict, GraphDatabase, SyntaxValidator, Tuple, basic_auth, df_reason):
    # Connect to Neo4j (required by CyVer)
    _uri: str = "bolt://localhost:7687"
    _auth: Tuple[str, str] = basic_auth("neo4j", "password-to-kg")
    _driver = GraphDatabase.driver(_uri, auth=_auth)

    _validator: SyntaxValidator = SyntaxValidator(_driver, check_multilabeled_nodes=False)

    def _validate_query(query_text: str) -> Dict[str, Any]:
        """Validate a single Cypher query using CyVer SyntaxValidator."""
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            return {"is_valid": False, "error": "Empty or None query"}
        try:
            # CyVer returns (is_valid: bool, metadata: dict)
            _is_valid: bool
            _metadata: Dict[str, Any]
            _is_valid, _metadata = _validator.validate(query_text.strip())
            return {"is_valid": _is_valid, "error": "" if _is_valid else str(_metadata)}
        except Exception as e:
            return {"is_valid": False, "error": f"Exception: {str(e)}"}

    # --- Process Reasoning ON Subset (up to 100) ---
    _count_re: int = min(100, len(df_reason))
    df_re_sample = df_reason.sample(n=_count_re, random_state=42).copy() if _count_re > 0 else df_reason.copy()

    _results_re = df_re_sample['query_text'].apply(_validate_query)
    df_re_sample['is_valid'] = _results_re.apply(lambda x: x['is_valid'])
    df_re_sample['validation_error'] = _results_re.apply(lambda x: x['error'])

    _driver.close()

    # Compute Statistics for Samples
    total_re: int = len(df_re_sample)
    valid_re: int = int(df_re_sample['is_valid'].sum())
    invalid_re: int = total_re - valid_re
    pct_invalid_re: float = (invalid_re / total_re * 100) if total_re > 0 else 0.0
    return df_re_sample, invalid_re, pct_invalid_re, total_re, valid_re


@app.cell
def _(
    invalid_re: int,
    mo,
    pct_invalid_re: float,
    total_re: int,
    valid_re: int,
):
    mo.md(f"""
    ### Validation Results Summary (Subset Sample: n=100)

    | Reasoning Mode | Total Sample | Valid Queries | Invalid Queries | % Invalid |
    | :--- | :--- | :--- | :--- | :--- |
    | **Reasoning ON** | {total_re} | {valid_re} | {invalid_re} | **{pct_invalid_re:.2f}%** |
    """)
    return


@app.cell
def _(df_re_sample, mo):
    # Sample invalid queries
    # Filter only rows with is_valid=False
    _invalid_on_df = df_re_sample[df_re_sample['is_valid'] == False]
    _invalid_on = _invalid_on_df[['question', 'query_text', 'validation_error']].head(10)

    _display = mo.vstack([
        mo.md("### Sample Invalid Queries: Reasoning ON"),
        mo.ui.table(_invalid_on) if not _invalid_on.empty else mo.md("_No invalid queries found._")
    ])
    _display
    return


@app.cell
def _(Any, Dict, List, ast, df_re_sample, re):
    def _clean_error(error_str: str) -> str:
        """Parse stringified list of errors and aggressively group by core category."""
        if not error_str or error_str == "Empty or None query":
            return error_str
        try:
            # Handle CyVer's string representation of list of dicts
            _data: List[Dict[str, Any]] = ast.literal_eval(error_str)
            if not _data or not isinstance(_data, list):
                return error_str
            _desc: str = _data[0].get('description', '')
            _desc_lower: str = _desc.lower()

            # Broad category matching
            if "invalid input" in _desc_lower or "extraneous input" in _desc_lower or "mismatched input" in _desc_lower:
                return "Syntax Error: Invalid or unexpected input"
            if "unknown function" in _desc_lower:
                return "Semantic Error: Unknown function"
            if "variable" in _desc_lower and "not defined" in _desc_lower:
                return "Semantic Error: Variable not defined"
            if "type mismatch" in _desc_lower:
                return "Semantic Error: Type mismatch"
            if "return can only be used at the end" in _desc_lower:
                return "Syntax Error: RETURN must be at query end"
            if "query cannot conclude with" in _desc_lower:
                return "Syntax Error: Query cannot conclude with clause"
            if "all sub queries in an union" in _desc_lower:
                return "Semantic Error: UNION return columns mismatch"
            if "missing" in _desc_lower and "expected" in _desc_lower:
                return "Syntax Error: Missing expected token"

            # Fallback: remove anything in double quotes (usually query extracts), then take text before colon
            _clean: str = re.sub(r'\(line \d+, column \d+ \(offset: \d+\)\)', '', _desc)
            _clean = re.sub(r'".*?"', '', _clean).strip()
            _clean = _clean.split(':')[0].strip().replace('\n', ' ')

            if _clean:
                return _clean[0].upper() + _clean[1:]
            return "Unknown Error Category"

        except (ValueError, SyntaxError, Exception):
            return error_str

    # Process Reasoning ON errors
    _err_re = df_re_sample[df_re_sample['is_valid'] == False]['validation_error'].apply(_clean_error)

    # Global ranking
    err_ranking = _err_re.value_counts().head(15)
    return (err_ranking,)


@app.cell
def _(err_ranking, mo, plt):
    # Ranking Chart of Common Syntax Errors
    if err_ranking.empty:
        _fig = mo.md("_No syntax errors found in the sample subsets._")
    else:
        _fig, _ax = plt.subplots(figsize=(14, 8))

        # Truncate long error descriptions so they fit in the figure margins
        _safe_index = [
            (str(lbl)[:75] + '...') if len(str(lbl)) > 75 else str(lbl)
            for lbl in err_ranking.index
        ]
        _plot_series = err_ranking.copy()
        _plot_series.index = _safe_index

        # Sort so most frequent is at the top
        _plot_series.iloc[::-1].plot(kind='barh', ax=_ax, color='salmon', alpha=0.9, edgecolor='darkred')

        _ax.set_title("Most Common Cypher Syntax Errors (Reasoning ON)")
        _ax.set_xlabel("Frequency (Sample n=200)")
        _ax.set_ylabel("Error Description (Cleaned)")

        # Add labels to the end of bars for clarity
        for i, v in enumerate(_plot_series.iloc[::-1]):
            _ax.text(v + 0.1, i, str(int(v)), color='darkred', va='center', fontweight='bold')

        _ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()

    _fig
    return


@app.cell
def _(invalid_re: int, plt, valid_re: int):
    # Pie Chart: Valid vs Invalid (Reasoning ON)
    _labels = ['Valid', 'Invalid']
    _sizes = [valid_re, invalid_re]
    _colors = ['teal', 'crimson']

    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.pie(_sizes, labels=_labels, autopct='%1.1f%%', startangle=140, colors=_colors)
    _ax.set_title('Cypher Query Syntax Validation (Reasoning ON)')

    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
