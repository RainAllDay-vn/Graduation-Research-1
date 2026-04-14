import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import os
    import re
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from model_evaluator import ModelEvaluator

    return ModelEvaluator, mo, os, pd, plt, re


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
    questions = evaluation_df['question']
    evaluation_df.head(10)
    return (questions,)


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

    def parse_cypher_query(response: str):
        """
        Parses a Cypher query using the canonical structure regex.
        Returns a dictionary of parts if matched, or a dictionary with an "error" key if invalid.
        """

        if response is None:
            return {"error": "Response is None"}

        cypher_pattern = re.compile(r"<cypher>(.*?)</cypher>", re.DOTALL | re.IGNORECASE)
        query = cypher_pattern.search(response.strip())
        if not query:
            return {"error": "No <cypher> tags found"}
        query = query.group(1)

        match = CYPHER_PARSER_REGEX.search(query.strip())
        if not match:
            return {"error": "Regex structure mismatch (canonical order ignored)"}

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
    ZERO_SHOT_PROMPT_TEMPLATE,
    evaluator: "ModelEvaluator",
    questions,
):
    section_1_evaluation_tasks = [
        (ZERO_SHOT_PROMPT_TEMPLATE, questions),
        (FEW_SHOT_PROMPT_TEMPLATE, questions),
    ]

    # Use the correct keyword argument 'input_data'
    section_1_responses = evaluator.call_model(input_data=section_1_evaluation_tasks)
    return (section_1_responses,)


@app.cell
def _(
    FEW_SHOT_PROMPT_TEMPLATE,
    ZERO_SHOT_PROMPT_TEMPLATE,
    mo,
    parse_cypher_query,
    pd,
    plt,
    questions,
    section_1_responses,
):
    def _show_summary():
        summary_results = []
        example_sections = []
        charts = []

        def create_pie_chart(valid, out_of_context, invalid, title):
            labels = ['Valid Response', 'Out of Context', 'Invalid Format']
            values = [valid, out_of_context, invalid]
            colors = ['#4CAF50', '#FF9800', '#f44336'] 

            # Filter out zero values to avoid clutter
            plot_labels = [l for l, v in zip(labels, values) if v > 0]
            plot_values = [v for v in values if v > 0]
            plot_colors = [c for c, v in zip(colors, values) if v > 0]

            if not plot_values:
                return mo.md(f"**{title}**: No data available")

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(plot_values, labels=plot_labels, autopct='%1.1f%%', startangle=90, colors=plot_colors)
            ax.set_title(title)
            plt.tight_layout()
            return fig

        for prompt_template, template_name in [
            (ZERO_SHOT_PROMPT_TEMPLATE, "Zero-Shot"),
            (FEW_SHOT_PROMPT_TEMPLATE, "Few-Shot"),
        ]:
            valid_count = 0
            out_of_context_count = 0
            invalid_format_count = 0

            valid_examples = []
            invalid_examples = []

            for question in questions:
                result = section_1_responses.get((prompt_template, question), {})
                response_text = result.get("response_text", "")
                context_length_exceeded = result.get("context_length_exceeded", 0)

                parsed_result = parse_cypher_query(response_text)
                is_valid = "error" not in parsed_result

                if context_length_exceeded == 1:
                    out_of_context_count += 1
                    error_msg = "Context length exceeded or empty response"
                    query_display = response_text
                elif not is_valid:
                    invalid_format_count += 1
                    error_msg = parsed_result.get("error", "Unknown parsing error")
                    query_display = response_text
                else:
                    valid_count += 1
                    error_msg = None
                    query_display = response_text

                # Collect examples
                if is_valid and len(valid_examples) < 2:
                    valid_examples.append({"question": question, "query": query_display})
                elif not is_valid and len(invalid_examples) < 2:
                    invalid_examples.append({"question": question, "query": query_display, "error": error_msg})

            summary_results.append({
                "Strategy": template_name,
                "Total": len(questions),
                "Valid": valid_count,
                "Out of Context": out_of_context_count,
                "Invalid Format": invalid_format_count,
                "Adherence": f"{(valid_count / len(questions)) * 100:.1f}%"
            })

            charts.append(create_pie_chart(valid_count, out_of_context_count, invalid_format_count, f"{template_name} Adherence"))

        summary_df = pd.DataFrame(summary_results)
        best_strategy = max(summary_results, key=lambda x: float(x["Adherence"].strip("%")))

        return mo.vstack([
            mo.md("## Evaluation Results: Query Adherence"),
            mo.hstack(charts, justify="space-around"),
            mo.md("### Numerical Summary"),
            summary_df,
            mo.md(f"**Conclusion:** The **{best_strategy['Strategy']}** strategy shows the best adherence to the canonical structure."),
        ])

    _show_summary()
    return


if __name__ == "__main__":
    app.run()
