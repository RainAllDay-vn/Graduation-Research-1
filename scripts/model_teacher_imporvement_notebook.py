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
