import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import sqlite3
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    return mo, os, pd, plt, sqlite3


@app.cell
def _(os, pd, sqlite3):
    _db_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'ai_cache.db')

    _conn = sqlite3.connect(_db_path)
    # Optimized: Moved JSON parsing, length calculation, and outlier filtering to SQL
    _query = """
    WITH parsed_cache AS (
        SELECT 
            json_extract(response, '$.content') AS content
        FROM ai_cache 
        WHERE model_name = 'Qwen/Qwen3.5-4B' 
          AND dataset_name = 'lc-quad-2.0' 
          AND include_reasoning = 0
          AND json_valid(response)
    )
    SELECT 
        content,
        LENGTH(content) AS response_length
    FROM parsed_cache
    """
    df = pd.read_sql_query(_query, _conn)
    _conn.close()
    return (df,)


@app.cell
def _(df, mo):
    # Stats markdown cell
    if df.empty:
        _stats_md = mo.md("# No data found for specified criteria.")
    else:
        _stats = df['response_length'].describe()
        _stats_md = mo.md(f"""
        # Summary Statistics: Qwen3.5-4B Response Lengths
        - **Total Responses:** {int(_stats['count'])}
        - **Mean Length:** {_stats['mean']:.2f} chars
        - **Median (50%):** {_stats['50%']:.2f} chars
        - **Standard Deviation:** {_stats['std']:.2f} chars
        - **Min Length:** {int(_stats['min'])} chars
        - **Max Length:** {int(_stats['max'])} chars
        - **25th Percentile:** {_stats['25%']:.2f} chars
        - **75th Percentile:** {_stats['75%']:.2f} chars
        """)
    _stats_md
    return

@app.cell
def _(df, plt):
    # Combined cell (middle 90%)
    _fig = None
    if not df.empty:
        # Calculate middle 90% bounds once
        _lower = df['response_length'].quantile(0.05)
        _upper = df['response_length'].quantile(0.95)
        _df_middle = df[(df['response_length'] >= _lower) & (df['response_length'] <= _upper)]

        # Create a figure with two subplots (Histogram and Box Plot)
        # sharex=True ensures both plots use the same scale for the x-axis
        _fig, (_ax_hist, _ax_box) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), 
                                                 gridspec_kw={'height_ratios': [3, 1]},
                                                 sharex=True)

        # 1. Histogram
        _ax_hist.hist(_df_middle['response_length'], bins=50, color='skyblue', edgecolor='black')
        _ax_hist.set_title(f"Distribution of Middle 90% Response Lengths\n(${_lower:.1f}$ to ${_upper:.1f}$ chars)")
        _ax_hist.set_ylabel("Frequency")
        _ax_hist.grid(axis='y', alpha=0.75)

        # 2. Box Plot
        _ax_box.boxplot(_df_middle['response_length'], vert=False, patch_artist=True, 
                        boxprops=dict(facecolor='lightgreen', color='green'),
                        medianprops=dict(color='red'))
        _ax_box.set_xlabel("Response Length (characters)")
        _ax_box.set_yticks([]) # Hide y-ticks for the box plot
        _ax_box.grid(axis='x', alpha=0.75)

        plt.tight_layout()

    _fig,
    return


if __name__ == "__main__":
    app.run()
