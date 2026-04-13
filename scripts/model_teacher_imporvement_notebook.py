import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import json
    import os
    import marimo as mo
    from model_evaluator import ModelEvaluator

    return ModelEvaluator, json, mo


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
def _(json):
    dataset_path = r"d:\Graduation-Research-1\dataset\mini\train.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        mini_dataset = json.load(f)
    return (mini_dataset,)


@app.cell
def _(ModelEvaluator, mini_dataset):
    # Extract questions for filtering
    _questions = [entry.get("question", entry.get("utterance", "")) for entry in mini_dataset]

    evaluator = ModelEvaluator(
        model_name="Qwen/Qwen3.5-4B", 
        api_key=""
    )

    # Fetch baseline responses from the teacher model (Qwen/Qwen3.5-4B with reasoning)
    baseline_df = evaluator.fetch_cached_responses(
        model_name="Qwen/Qwen3.5-4B",
        dataset_name="lc-quad-2.0",
        include_reasoning=1,
        questions = _questions
    )
    return (baseline_df,)


@app.cell
def _(baseline_df, mini_dataset, mo):
    mo.md(f"""
    Loaded **{len(mini_dataset)}** entries from the mini dataset.
    Found **{len(baseline_df)}** cached baseline responses.
    """)
    return


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
    mo.md(
        r"""
        ## 1. Structured Query Generation

        Generating graph queries like Cypher requires high precision. Even small deviations in syntax or structure can lead to execution failures. Structured Query Generation aims to enforce a strict format on the model's output to ensure it is consistently parseable and syntactically correct.

        ### Why Structure Matters
        When we use LLMs as "Teacher Models" to generate synthetic data or fine-tuning targets, we need to guarantee that the output can be automatically processed. By enforcing structure, we:
        - **Decrease Parsing Errors**: Eliminating the need for complex regex or manual cleaning.
        - **Improve Query Validity**: Constraining the model to follow specific Cypher patterns (e.g., proper `MATCH` ... `RETURN` flow).
        - **Standardize Metadata**: Ensuring that reasoning steps and final queries are clearly separated.

        In this section, we will define a structured prompt template and evaluate how well the model adheres to these constraints compared to our baseline.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Defining the Canonical Cypher Structure
        To ensure the model produces predictable and valid Cypher queries for our **Q&A system**, we enforce a strict read-only sequence. This prevents the model from generating write operations and ensures maximum parseability:
        1.  **READ**: `MATCH` or `OPTIONAL MATCH`
        2.  **FILTER**: `WHERE` (immediately following the MATCH)
        3.  **RETURN**: `RETURN`
        4.  **POST-PROCESS**: `ORDER BY` or `LIMIT`
        """
    )
    return


@app.cell
def _():
    # Canonical order of Cypher clauses for validation and prompt engineering
    CANONICAL_CYPHER_ORDER = [
        "MATCH",
        "OPTIONAL MATCH",
        "WHERE",
        "RETURN",
        "ORDER BY",
        "LIMIT",
    ]

    def _get_structured_query_prompt_snippet():
        """Returns a prompt snippet that enforces the Cypher structure."""
        order_str = " -> ".join(CANONICAL_CYPHER_ORDER)
        return f"CRITICAL: Always follow the standard Cypher clause order: {order_str}. Do not skip to RETURN before MATCHing patterns."

    _get_structured_query_prompt_snippet()

    return CANONICAL_CYPHER_ORDER


if __name__ == "__main__":
    app.run()
