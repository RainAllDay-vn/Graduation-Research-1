import marimo

__generated_with = "0.21.1"
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

    return mo, os, pd, plt, sqlite3


@app.cell
def _(mo):
    mo.md("""
    # Model Analysis: Reasoning Performance Comparison

    This notebook analyzes and compares the performance of **Qwen/Qwen3.5-4B** on the **LC-QuAD 2.0** dataset, specifically focusing on the difference between responses generated with and without reasoning enabled.

    ### Key Objectives:
    - Compare Cypher query character lengths.
    - Analyze the "thinking" process length for reasoning-enabled models.
    - Visualize distributions and identify potential outliers or patterns.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data Loading and Preparation

    We retrieve cached model responses from a SQLite database (`ai_cache.db`).
    - For **Reasoning OFF**, we extract the JSON content directly.
    - For **Reasoning ON**, we parse the content to separate the `<think>` reasoning block from the final Cypher query.
    """)
    return


@app.cell
def _(os, pd, sqlite3):
    _db_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'ai_cache.db')
    _conn = sqlite3.connect(_db_path)

    # Load reasoning=0 data with all fields
    _query_no = """
    SELECT 
        *,
        json_extract(response, '$.content') AS content
    FROM ai_cache 
    WHERE model_name = 'Qwen/Qwen3.5-4B' 
      AND dataset_name = 'lc-quad-2.0' 
      AND include_reasoning = 0
      AND json_valid(response)
    """
    df_no_reason = pd.read_sql_query(_query_no, _conn)
    df_no_reason['query_length'] = pd.to_numeric(df_no_reason['content'].str.len(), errors='coerce')

    # Load reasoning=1 data with all fields
    _query_re = """
    SELECT 
        *,
        json_extract(response, '$.content') AS full_content
    FROM ai_cache 
    WHERE model_name = 'Qwen/Qwen3.5-4B' 
      AND dataset_name = 'lc-quad-2.0' 
      AND include_reasoning = 1
      AND json_valid(response)
    """
    df_reason = pd.read_sql_query(_query_re, _conn)

    def _extract_parts(text):
        if not text:
            return "", ""
        # The model response for reasoning models usually wraps thinking in <think>...</think>
        # We look for the closing tag to separate thinking from the final query.
        _clean_text = text.replace("<think>", "")
        before, sep, after = _clean_text.partition("</think>")
        if sep:
            return before.strip(), after.strip()
        return None

    _parts = df_reason['full_content'].apply(_extract_parts)
    invalid_response_count = int(_parts.isna().sum())
    _valid_parts = _parts.dropna()

    # Initialize columns with empty/None
    df_reason['thinking_text'] = ""
    df_reason['query_text'] = ""

    # Map valid parts back to the dataframe
    if not _valid_parts.empty:
        df_reason.loc[_valid_parts.index, 'thinking_text'] = _valid_parts.apply(lambda x: x[0])
        df_reason.loc[_valid_parts.index, 'query_text'] = _valid_parts.apply(lambda x: x[1])

    df_reason['thinking_length'] = pd.to_numeric(df_reason['thinking_text'].str.len(), errors='coerce')
    df_reason['query_length'] = pd.to_numeric(df_reason['query_text'].str.len(), errors='coerce')

    _conn.close()
    return df_no_reason, df_reason, invalid_response_count


@app.cell
def _(mo):
    mo.md("""
    ## Data Preview
    """)
    return


@app.cell
def _(df_no_reason, df_reason, mo):
    _preview = mo.vstack([
        mo.md("### Preview: Reasoning OFF"),
        mo.ui.table(df_no_reason.head()),
        mo.md("### Preview: Reasoning ON"),
        mo.ui.table(df_reason.head())
    ])
    _preview
    return


@app.cell
def _(df_no_reason, df_reason, invalid_response_count, mo):
    if df_no_reason.empty or df_reason.empty:
        _stats_md = mo.md("# Missing data for comparison.")
    else:
        _s_no = df_no_reason['query_length'].describe()
        _s_re = df_reason['query_length'].describe()
        _s_th = df_reason['thinking_length'].describe()

        # Create a small warning string if there are invalid responses
        _invalid_msg = ""
        if invalid_response_count > 0:
            _invalid_msg = f"**Note:** {invalid_response_count} responses were skipped because they were missing `</think>` tags."

        _stats_md = mo.md(f"""
        ## Summary Statistics: Reasoning OFF vs ON

        This table compares the **Cypher Query** character lengths:
        - **OFF: Response**: The full response (query only) when reasoning is disabled.
        - **ON: Query**: The extracted query section when reasoning is enabled.
        - **ON: Thinking**: The reasoning content extracted from `<think>` tags.

        | Metric | OFF: Response | ON: Query | ON: Thinking |
        | :--- | :--- | :--- | :--- |
        | **Count** | {_s_no['count']:.0f} | {_s_re['count']:.0f} | {_s_th['count']:.0f} |
        | **Mean** | {_s_no['mean']:.2f} | {_s_re['mean']:.2f} | {_s_th['mean']:.2f} |
        | **Median (50%)** | {_s_no['50%']:.2f} | {_s_re['50%']:.2f} | {_s_th['50%']:.2f} |
        | **Std Dev** | {_s_no['std']:.2f} | {_s_re['std']:.2f} | {_s_th['std']:.2f} |
        | **Min** | {_s_no['min']:.0f} | {_s_re['min']:.0f} | {_s_th['min']:.0f} |
        | **Max** | {_s_no['max']:.0f} | {_s_re['max']:.0f} | {_s_th['max']:.0f} |

        {_invalid_msg}
        """)

    _stats_md
    return


@app.cell
def _(df_no_reason, df_reason, mo, pd):
    def _get_samples(df, text_col, n=5):
        if df.empty:
            return pd.DataFrame()

        # Ensure variety by dropping identical queries first
        _unique_df = df.drop_duplicates(subset=[text_col])
        _n = min(n, len(_unique_df))

        # Random sample for a diverse view
        return _unique_df.sample(n=_n, random_state=42)

    _samples_off = _get_samples(df_no_reason, 'content')
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

    ### Reasoning OFF (Average: {df_no_reason['query_length'].mean() if not df_no_reason.empty else 0:.1f} chars)
    {_format_list(_samples_off, 'content')}

    ### Reasoning ON (Average: {df_reason['query_length'].mean() if not df_reason.empty else 0:.1f} chars)
    {_format_list(_samples_on, 'query_text')}
    """)

    _content
    return


@app.cell
def _(mo):
    mo.md("""
    ## Distribution Analysis

    The following visualizations compare the character length distributions of the generated Cypher queries.
    Note that for reasoning-enabled models, we only measure the *final query* length, excluding the thinking process.
    """)
    return


@app.cell
def _(df_no_reason, df_reason, plt):
    _fig = None
    if not df_no_reason.empty and not df_reason.empty:
        _l0, _u0 = df_no_reason['query_length'].quantile(0.05), df_no_reason['query_length'].quantile(0.95)
        _l1, _u1 = df_reason['query_length'].quantile(0.05), df_reason['query_length'].quantile(0.95)

        _shared_min = min(_l0, _l1)
        _shared_max = max(_u0, _u1)

        _fig, _ax = plt.subplots(figsize=(10, 6))

        _ax.hist(df_no_reason['query_length'], bins=50, range=(_shared_min, _shared_max), 
                 alpha=0.5, label='Reasoning OFF (Full Response)', color='blue', edgecolor='darkblue')
        _ax.hist(df_reason['query_length'], bins=50, range=(_shared_min, _shared_max), 
                 alpha=0.5, label='Reasoning ON (Extracted Query)', color='orange', edgecolor='darkorange')

        _ax.set_title("Query Length Distribution: Reasoning OFF vs ON")
        _ax.set_xlabel("Query Length (characters)")
        _ax.set_ylabel("Frequency")
        _ax.legend()
        _ax.grid(True, alpha=0.3)
        plt.tight_layout()

    _fig
    return


@app.cell
def _(df_no_reason, df_reason, plt):
    _fig = None
    if not df_no_reason.empty and not df_reason.empty:
        # We still calculate the upper bound for the shared axis
        _u0 = df_no_reason['query_length'].quantile(0.95)
        _u1 = df_reason['query_length'].quantile(0.95)
        _shared_max = max(_u0, _u1)

        _fig, _ax = plt.subplots(figsize=(10, 4))
        _data = [df_no_reason['query_length'], df_reason['query_length']]
        _ax.boxplot(_data, vert=False, labels=['Reasoning OFF (Full)', 'Reasoning ON (Query)'], 
                    patch_artist=True)

        # Change the lower limit to 0
        _ax.set_xlim(0, _shared_max) 

        _ax.set_title("Response Length Box Comparison")
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


if __name__ == "__main__":
    app.run()
