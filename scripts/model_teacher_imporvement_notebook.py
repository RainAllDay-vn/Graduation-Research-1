import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import os
    import re
    import marimo as mo
    import pandas as pd
    from model_evaluator import ModelEvaluator

    return ModelEvaluator, mo, os, pd, re


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Setting Up

    Before testing our improvement techniques, we initialize the environment and load a sampled "mini" dataset.

    ### Why a Mini Dataset?
    We use a subset of **100 random entries** from the LC-QuAD 2.0 dataset (`dataset/mini/train.json`) to:
    1. **Speed up Iteration**: Faster inference times during prompt engineering and debugging.
    2. **Reduce Costs**: Minimize API usage for large-scale teacher model responses.
    3. **Focused Testing**: Quickly identify if a specific constraint or reasoning pattern is being followed correctly.
    """)
    return


@app.cell
def _(pd):
    dataset_path = "dataset/medquad/data.parquet"
    evaluation_df = pd.read_parquet(dataset_path).sample(100, random_state=42)[['question', 'answer']]
    evaluation_df.head(10)
    return (evaluation_df,)


@app.cell
def _(ModelEvaluator, os):
    from dotenv import load_dotenv
    load_dotenv()

    evaluator: ModelEvaluator = ModelEvaluator(
        model_name="Qwen/Qwen3.5-4B", 
        api_key=os.getenv("LITELLM_API_KEY", "")
    )
    return (evaluator,)


@app.cell
def _(mo):
    mo.md(r"""
    # Model Teacher Improvement: Enhancing Query Generation

    To improve the efficiency and reliability of the cypher query generation, we will test the following methods:

    ### 1. Enforce Structured Query Generation
    Standardize the model's output format to ensure maximum parseability:
    - **Strict Query Format:** Make components of Cypher query (like MATCH, RETURN, ...) follow a strict order to allow better parsing.

    ### 2. Optimize Reasoning Efficiency
    Reduce the character length of the reasoning process (`<think>` block) without compromising query quality. This can be achieved through:
    - **Prompt Engineering:** Refining instructions to encourage "concise but logical" derivations.

    ### 3. Minimize Error Frequency
    Address common syntax and semantic errors discovered in the current benchmarks:
    - **Prompt Engineering:** Add examples or guidelines to avoid common errors.
    - **Self-Correction:** Allow the LLM to retry if the query is invalid.
    - **Ontology Awareness:** Injecting more specific medical ontology constraints into the system prompt.
    - **Error-Correcting Fine-Tuning:** Training on a "synthetic correction" dataset where the model learns to fix its own previous mistakes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Structured Query Generation

    Generating graph queries like Cypher requires high precision. Even small deviations in syntax or structure can lead to execution failures. Structured Query Generation aims to enforce a strict format on the model's output to ensure it is consistently parseable and syntactically correct.

    ### Why Structure Matters
    When we use LLMs as "Teacher Models" to generate synthetic data or fine-tuning targets, we need to guarantee that the output can be automatically processed. By enforcing structure, we:
    - **Decrease Parsing Errors**: Eliminating the need for complex regex or manual cleaning.
    - **Improve Query Validity**: Constraining the model to follow specific Cypher patterns (e.g., proper `MATCH` ... `RETURN` flow).

    In this section, we will define a structured prompt template and evaluate how well the model adheres to these constraints compared to our baseline.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Defining the Canonical Cypher Structure
    To ensure the model produces predictable and valid Cypher queries for our **Q&A system**, we enforce a strict read-only sequence. This prevents the model from generating write operations and ensures maximum parseability:
    1.  **READ**: `MATCH` or `OPTIONAL MATCH`
    2.  **FILTER**: `WHERE` (immediately following the MATCH)
    3.  **RETURN**: `RETURN`
    4.  **POST-PROCESS**: `ORDER BY` or `LIMIT`
    """)
    return


@app.cell
def _(mo):
    # Canonical order of Cypher clauses for validation and prompt engineering
    CANONICAL_CYPHER_ORDER = [
        "MATCH",
        "OPTIONAL MATCH",
        "WHERE",
        "RETURN",
        "ORDER BY",
        "LIMIT",
    ]

    def get_structured_query_prompt_snippet():
        order_str = " -> ".join(CANONICAL_CYPHER_ORDER)
        return mo.md(f"CRITICAL: Always follow the standard Cypher clause order: `{order_str}`. Do not skip to RETURN before MATCHING patterns.")

    get_structured_query_prompt_snippet()
    return (get_structured_query_prompt_snippet,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Robust Query Parsing with Regex
    With our strict structure, we can now use a predictable regular expression to decompose the generated queries into their constituent parts. This is useful for validation, logging, and performance analysis.
    """)
    return


@app.cell
def _(mo, re):
    # Regex designed to capture the structural components of our canonical Cypher format
    # It supports: (OPTIONAL MATCH/MATCH) -> (WHERE)? -> (RETURN) -> (ORDER BY)? -> (LIMIT)?
    CYPHER_PARSER_REGEX = re.compile(
        r"(?i)"  # Case-insensitive
        r"(?P<match_type>OPTIONAL\s+MATCH|MATCH)\s+(?P<match_clause>.+?)"
        r"(?:\s+WHERE\s+(?P<where_clause>.+?))?"
        r"\s+RETURN\s+(?P<return_clause>.+?)"
        r"(?:\s+ORDER\s+BY\s+(?P<order_clause>.+?))?"
        r"(?:\s+LIMIT\s+(?P<limit_clause>\d+))?"
        r"\s*;?$",  # Optional semicolon and trailing whitespace
        re.DOTALL | re.MULTILINE
    )

    def parse_cypher_query(query: str):
        """
        Parses a Cypher query using the canonical structure regex.
        Returns a dictionary of parts if matched, or None if invalid.
        """
        match = CYPHER_PARSER_REGEX.search(query.strip())
        if not match:
            return None

        return {k: v.strip() if v else None for k, v in match.groupdict().items()}

    # Example test
    _test_query = "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s) WHERE s.name = 'Cough' RETURN d.name ORDER BY d.name LIMIT 5"
    _parsed = parse_cypher_query(_test_query)

    mo.md(
        f"""
        #### Parser Demonstration
        **Input Query:**
        ```cypher
        {_test_query}
        ```

        **Parsed Components:**
        {mo.as_html(_parsed) if _parsed else "❌ Failed to parse"}
        """
    )
    return (parse_cypher_query,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Approach 1: Zero-Shot Strict Guidance

    In this approach, we provide explicit rules dictating the structure to the model without giving examples. We define exactly what is allowed and what is forbidden.
    """)
    return


@app.cell
def _(get_structured_query_prompt_snippet, mo):
    # Base Zero-Shot Prompt Template
    ZERO_SHOT_PROMPT_TEMPLATE = f"""You are a strict Cypher query generator for a read-only Question Answering system.
    Your task is to convert the given natural language question into a Cypher query.

    {get_structured_query_prompt_snippet()}

    Additional Constraints:
    1. DO NOT use these clauses: WITH, UNWIND, CREATE, MERGE, SET, DELETE.
    2. Separate your reasoning from the final query.
    3. Wrap your final, executable Cypher query inside `<cypher>` and `</cypher>` tags.

    Question: {{question}}
    """

    mo.md(f'''
    ```
    {ZERO_SHOT_PROMPT_TEMPLATE}
    ```
    ''')
    return (ZERO_SHOT_PROMPT_TEMPLATE,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Approach 2: Few-Shot Demonstrations

    Here, we reinforce the strict structure by providing a few concrete examples showing the exact input -> reasoning -> `<cypher>` output pipeline we expect.
    """)
    return


@app.cell
def _(ZERO_SHOT_PROMPT_TEMPLATE, mo):
    # Few-Shot Demonstrations (extending the Zero Shot template)
    FEW_SHOT_PROMPT_TEMPLATE = ZERO_SHOT_PROMPT_TEMPLATE + """
    ---
    EXAMPLES:

    Question: What is a symptom of asthma?
    Reasoning: I need to match the Disease node representing "asthma", find diseases that have a symptom relationship to a Symptom node, and return the symptom name.
    <cypher>
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE d.name = 'asthma'
    RETURN s.name
    </cypher>

    Question: List 5 diseases associated with a cough, ordered alphabetically.
    Reasoning: I need to match a Symptom node named 'cough', traverse back to Disease nodes via the HAS_SYMPTOM relationship, return the disease names, order them alphabetically, and limit the result to 5.
    <cypher>
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name = 'cough'
    RETURN d.name
    ORDER BY d.name
    LIMIT 5
    </cypher>
    ---

    Question: {{question}}
    """

    mo.md(f'''
    ```
    {FEW_SHOT_PROMPT_TEMPLATE}
    ```
    ''')
    return (FEW_SHOT_PROMPT_TEMPLATE,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Running the Evaluation
    """)
    return


@app.cell
def _(
    FEW_SHOT_PROMPT_TEMPLATE,
    evaluation_df,
    evaluator,
    mo,
    parse_cypher_query,
    pd,
    re,
):
    def _run_evaluation():
        _cypher_pattern = re.compile(r"<cypher>(.*?)</cypher>", re.DOTALL | re.IGNORECASE)

        _questions = evaluation_df['question'].tolist()
        _input = [
            (FEW_SHOT_PROMPT_TEMPLATE.replace("{{question}}", q), q) 
            for q in _questions
        ]

        evaluator.call_model(_input)

        _cached_responses = evaluator.fetch_all_cached_responses(FEW_SHOT_PROMPT_TEMPLATE, _questions, evaluator.include_reasoning)

        _results = []
        for _q in _questions:
            _response = next((r for r in _cached_responses if r.get('question') == _q), {})
            _response_text = _response.get('response_text', '')

            _cypher_match = _cypher_pattern.search(_response_text)
            _extracted_cypher = _cypher_match.group(1).strip() if _cypher_match else ''

            if _extracted_cypher:
                _parsed = parse_cypher_query(_extracted_cypher)
                if _parsed:
                    _results.append({
                        'question': _q,
                        'status': 'parsed',
                        'match_clause': _parsed.get('match_clause'),
                        'where_clause': _parsed.get('where_clause'),
                        'return_clause': _parsed.get('return_clause'),
                    })
                else:
                    _results.append({
                        'question': _q,
                        'status': 'invalid_structure',
                        'extracted_cypher': _extracted_cypher[:100],
                    })
            else:
                _results.append({
                    'question': _q,
                    'status': 'no_cypher_tag',
                    'response_preview': _response_text[:100],
                })

        _results_df = pd.DataFrame(_results)
        _status_counts = _results_df['status'].value_counts()

        _parsed_count = _status_counts.get('parsed', 0)
        _invalid_count = _status_counts.get('invalid_structure', 0)
        _no_tag_count = _status_counts.get('no_cypher_tag', 0)
        _total = len(_results_df)

        return mo.vstack([
            mo.md(f"### Evaluation Results ({_total} questions)"),
            mo.md(f"- **Successfully Parsed**: {_parsed_count} ({_parsed_count/_total*100:.1f}%)"),
            mo.md(f"- **Invalid Structure**: {_invalid_count} ({_invalid_count/_total*100:.1f}%)"),
            mo.md(f"- **No Cypher Tag**: {_no_tag_count} ({_no_tag_count/_total*100:.1f}%)"),
            mo.md("#### Sample Results:"),
            _results_df.head(10)
        ])

    _run_evaluation()
    return


if __name__ == "__main__":
    app.run()
