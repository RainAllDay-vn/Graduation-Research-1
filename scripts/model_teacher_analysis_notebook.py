import marimo

__generated_with = "0.22.0"
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
        np,
        pd,
        plt,
        re,
    )


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
def _(ModelEvaluator, pd):
    evaluator = ModelEvaluator(model_name='dummy', api_key='')

    # Load reasoning=0 data with all fields
    df_no_reason = evaluator.fetch_cached_responses('Qwen/Qwen3.5-4B', 'lc-quad-2.0', include_reasoning=0)
    df_no_reason['query_length'] = pd.to_numeric(df_no_reason['content'].str.len(), errors='coerce')

    # Load reasoning=1 data with all fields
    df_reason = evaluator.fetch_cached_responses('Qwen/Qwen3.5-4B', 'lc-quad-2.0', include_reasoning=1)
    df_reason.rename(columns={'content': 'full_content'}, inplace=True)

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
        df_reason = df_reason.loc[_valid_parts.index].copy()
        df_reason['thinking_text'] = _valid_parts.apply(lambda x: x[0])
        df_reason['query_text'] = _valid_parts.apply(lambda x: x[1])

    df_reason['thinking_length'] = pd.to_numeric(df_reason['thinking_text'].str.len(), errors='coerce')
    df_reason['query_length'] = pd.to_numeric(df_reason['query_text'].str.len(), errors='coerce')
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


@app.cell
def _(mo):
    mo.md("""
    ## Cypher Query Syntax Validation (CyVer AST)

    We validate all generated Cypher queries using **CyVer**'s `SyntaxValidator`, which uses the openCypher ANTLR grammar under the hood to perform real AST-based syntax validation.

    This analysis determines what percentage of the model's generated queries are syntactically invalid.
    """)
    return


@app.cell
def _(
    Any,
    Dict,
    GraphDatabase,
    SyntaxValidator,
    Tuple,
    basic_auth,
    df_no_reason,
    df_reason,
):
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

    # --- Process Reasoning OFF Subset (up to 200) ---
    _count_no: int = min(200, len(df_no_reason))
    df_no_sample = df_no_reason.sample(n=_count_no, random_state=42).copy() if _count_no > 0 else df_no_reason.copy()

    _results_no = df_no_sample['content'].apply(_validate_query)
    df_no_sample['is_valid'] = _results_no.apply(lambda x: x['is_valid'])
    df_no_sample['validation_error'] = _results_no.apply(lambda x: x['error'])

    # --- Process Reasoning ON Subset (up to 200) ---
    _count_re: int = min(200, len(df_reason))
    df_re_sample = df_reason.sample(n=_count_re, random_state=42).copy() if _count_re > 0 else df_reason.copy()

    _results_re = df_re_sample['query_text'].apply(_validate_query)
    df_re_sample['is_valid'] = _results_re.apply(lambda x: x['is_valid'])
    df_re_sample['validation_error'] = _results_re.apply(lambda x: x['error'])

    _driver.close()

    # Compute Statistics for Samples
    total_no: int = len(df_no_sample)
    valid_no: int = int(df_no_sample['is_valid'].sum())
    invalid_no: int = total_no - valid_no
    pct_invalid_no: float = (invalid_no / total_no * 100) if total_no > 0 else 0.0

    total_re: int = len(df_re_sample)
    valid_re: int = int(df_re_sample['is_valid'].sum())
    invalid_re: int = total_re - valid_re
    pct_invalid_re: float = (invalid_re / total_re * 100) if total_re > 0 else 0.0
    return (
        df_no_sample,
        df_re_sample,
        invalid_no,
        invalid_re,
        pct_invalid_no,
        pct_invalid_re,
        total_no,
        total_re,
        valid_no,
        valid_re,
    )


@app.cell
def _(
    invalid_no: int,
    invalid_re: int,
    mo,
    pct_invalid_no: float,
    pct_invalid_re: float,
    total_no: int,
    total_re: int,
    valid_no: int,
    valid_re: int,
):
    mo.md(f"""
    ### Validation Results Summary (Subset Sample: n=200 per group)

    | Reasoning Mode | Total Sample | Valid Queries | Invalid Queries | % Invalid |
    | :--- | :--- | :--- | :--- | :--- |
    | **Reasoning OFF** | {total_no} | {valid_no} | {invalid_no} | **{pct_invalid_no:.2f}%** |
    | **Reasoning ON** | {total_re} | {valid_re} | {invalid_re} | **{pct_invalid_re:.2f}%** |
    """)
    return


@app.cell
def _(df_no_sample, df_re_sample, mo):
    # Sample invalid queries for both
    # Filter only rows with is_valid=False
    _invalid_off_df = df_no_sample[df_no_sample['is_valid'] == False]
    _invalid_on_df = df_re_sample[df_re_sample['is_valid'] == False]

    _invalid_off = _invalid_off_df[['question', 'content', 'validation_error']].head(10)
    _invalid_on = _invalid_on_df[['question', 'query_text', 'validation_error']].head(10)

    _display = mo.vstack([
        mo.md("### Sample Invalid Queries: Reasoning OFF"),
        mo.ui.table(_invalid_off) if not _invalid_off.empty else mo.md("_No invalid queries found._"),
        mo.md("### Sample Invalid Queries: Reasoning ON"),
        mo.ui.table(_invalid_on) if not _invalid_on.empty else mo.md("_No invalid queries found._")
    ])
    _display
    return


@app.cell
def _(Any, Dict, List, ast, df_no_sample, df_re_sample, pd, re):
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

    # Process Reasoning OFF errors
    _err_no = df_no_sample[df_no_sample['is_valid'] == False]['validation_error'].apply(_clean_error)
    # Process Reasoning ON errors
    _err_re = df_re_sample[df_re_sample['is_valid'] == False]['validation_error'].apply(_clean_error)

    # Combine for global ranking
    err_ranking = pd.concat([_err_no, _err_re]).value_counts().head(15)
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

        _ax.set_title("Most Common Cypher Syntax Errors (Top 15 Pattern Ranking)")
        _ax.set_xlabel("Frequency (Combined Sample n=400)")
        _ax.set_ylabel("Error Description (Cleaned)")

        # Add labels to the end of bars for clarity
        for i, v in enumerate(_plot_series.iloc[::-1]):
            _ax.text(v + 0.1, i, str(int(v)), color='darkred', va='center', fontweight='bold')

        _ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()

    _fig
    return


@app.cell
def _(invalid_no: int, invalid_re: int, np, plt, valid_no: int, valid_re: int):
    # Grouped Bar Chart: Valid vs Invalid Comparison (Sampled Subset)
    _categories: list[str] = ['Reasoning OFF', 'Reasoning ON']
    _valid_counts: list[int] = [valid_no, valid_re]
    _invalid_counts: list[int] = [invalid_no, invalid_re]

    _x: np.ndarray = np.arange(len(_categories))
    _width: float = 0.35

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.bar(_x - _width/2, _valid_counts, _width, label='Valid', color='teal', alpha=0.8)
    _ax.bar(_x + _width/2, _invalid_counts, _width, label='Invalid', color='crimson', alpha=0.8)

    _ax.set_ylabel('Count')
    _ax.set_title('Cypher Query Syntax Validation Comparison')
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_categories)
    _ax.legend()
    _ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    df_no_reason,
    df_reason,
    mo,
    pct_invalid_no: float,
    pct_invalid_re: float,
    total_no: int,
    total_re: int,
    valid_no: int,
    valid_re: int,
):
    _avg_len_no = df_no_reason['query_length'].mean() if not df_no_reason.empty else 0
    _avg_len_re = df_reason['query_length'].mean() if not df_reason.empty else 0
    _avg_think_len = df_reason['thinking_length'].mean() if not df_reason.empty else 0

    # Calculate the improvement or degradation in validity
    _diff_validity = pct_invalid_no - pct_invalid_re
    _validity_msg = ""
    if _diff_validity > 0:
        _validity_msg = f"Reasoning ON improved syntax validity by **{_diff_validity:.2f}%**."
    elif _diff_validity < 0:
        _validity_msg = f"Reasoning ON actually increased syntax errors by **{abs(_diff_validity):.2f}%**."
    else:
        _validity_msg = "Reasoning mode had no impact on syntax validity percentage."


    mo.md(f"""
    ## Conclusion: Key Discoveries

    Based on the analysis of **Qwen/Qwen3.5-4B** on the **LC-QuAD 2.0** dataset, we have observed the following:

    ### 1. Structural Complexity
    - **Query Length:** Enabling reasoning leads to an average query length of **{_avg_len_re:.1f}** characters, compared to **{_avg_len_no:.1f}** characters when reasoning is disabled. 
    - **Thinking Overhead:** The model generates an average of **{_avg_think_len:.1f}** characters of internal reasoning before outputting the final Cypher query.

    ### 2. Syntactic Reliability (Subset n=200)
    - **Reasoning OFF:** Successfully generated **{valid_no}** valid queries out of {total_no} (**{100-pct_invalid_no:.2f}%** success rate).
    - **Reasoning ON:** Successfully generated **{valid_re}** valid queries out of {total_re} (**{100-pct_invalid_re:.2f}%** success rate).
    - **Impact:** {_validity_msg}

    ### Final Takeaway
    The data suggests that while the internal reasoning process (thinking) adds significant token overhead, it {"significantly" if abs(_diff_validity) > 5 else "slightly"} affects the grammatical correctness of the resulting Cypher. This analysis serves as a baseline for further ontology-constrained fine-tuning.
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    ## Next Steps & Future Work

    To improve the efficiency and reliability of the cypher query generation, the following objectives have been identified:

    ### 1. Optimize Reasoning Efficiency
    Reduce the character length of the reasoning process (`<think>` block) without compromising query quality. This can be achieved through:
    - **Prompt Engineering:** Refining instructions to encourage "concise but logical" derivations.

    ### 2. Minimize Error Frequency
    Address common syntax and semantic errors discovered in the current benchmarks:
    - **Prompt Engineering:** Add examples or guidelines to avoid common errors.
    - **Self-Correction:** Allow the LLM to retry if the query is invalid.
    - **Ontology Awareness:** Injecting more specific medical ontology constraints into the system prompt.
    - **Error-Correcting Fine-Tuning:** Training on a "synthetic correction" dataset where the model learns to fix its own previous mistakes.

    ### 3. Enforce Structured Query Generation
    Standardize the model's output format to ensure maximum parseability:
    - **Strict Query Format:** Make components of Cypher query (like MATCH, RETURN, ...) follow a strict order to allow better parsing.
    """)
    return


if __name__ == "__main__":
    app.run()
