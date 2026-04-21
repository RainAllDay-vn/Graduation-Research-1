# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "litellm",
#     "neo4j",
#     "python-dotenv",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure the root directory is in sys.path
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_file_dir, "..", ".."))
    if _root not in sys.path:
        sys.path.append(_root)
    os.chdir(_root)

    from app.data_loader.medquad_data_loader import MedquadDataLoader
    from app.knowledge_graph import KnowledgeGraph
    from app.request_repository import RequestRepository
    from app.llm_client import LlmClient
    from app.question_to_query_pipeline import QuestionToQueryPipeline
    from app.validator import validate_query

    return (
        KnowledgeGraph,
        LlmClient,
        MedquadDataLoader,
        QuestionToQueryPipeline,
        RequestRepository,
        mo,
        pd,
        plt,
        validate_query,
    )


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
    - **Ontology Awareness:** Injecting more specific medical ontology constraints into the system prompt.
    """)
    return


@app.cell
def _(
    KnowledgeGraph,
    LlmClient,
    MedquadDataLoader,
    QuestionToQueryPipeline,
    RequestRepository,
):
    loader = MedquadDataLoader()
    kg = KnowledgeGraph(data_loader=loader)
    llm_client = LlmClient()
    repository = RequestRepository()
    pipeline = QuestionToQueryPipeline(
        llm_client=llm_client,
        request_repository=repository,
        knowledge_graph=kg
    )
    return (loader,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Setting Up

    Before testing our improvement techniques, we initialize the environment and load a sampled "mini" dataset of 100 random entries from the Medquad dataset.
    """)
    return


@app.cell
def _(loader, pd):
    _dataset = loader.load_dataset()
    _df = pd.DataFrame(_dataset, columns=['question', 'answer'])
    _evaluation_df = _df.sample(100, random_state=42)
    questions_and_answers = list(zip(_evaluation_df['question'], _evaluation_df['answer']))
    return (questions_and_answers,)


@app.cell
def _(mo, plt, validate_query):
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

    def evaluate_performance(repository, pipeline_run_request, template_name, kg):
        valid_count = 0
        out_of_context_count = 0
        invalid_format_count = 0

        total_reasoning_chars = 0
        count_with_reasoning = 0

        questions_count = len(pipeline_run_request.data)

        for model_request in pipeline_run_request.to_model_request():
            cached_request = repository.get_request_by_metadata(model_request)
            if not cached_request:
                continue

            response_text = cached_request.response or ""
            reasoning_content = cached_request.reasoning or ""
            context_length_exceeded = cached_request.context_length_exceeded or False

            if reasoning_content:
                total_reasoning_chars += len(reasoning_content)
                count_with_reasoning += 1

            validation_result = validate_query(kg, response_text)
            is_valid = validation_result == "OK"

            if context_length_exceeded:
                out_of_context_count += 1
            elif not is_valid:
                invalid_format_count += 1
            else:
                valid_count += 1

        avg_reasoning_len = round(total_reasoning_chars / count_with_reasoning) if count_with_reasoning > 0 else 0

        metrics = {
            "Strategy": template_name,
            "Total": questions_count,
            "Valid": valid_count,
            "Valid %": f"{(valid_count / questions_count) * 100:.1f}%" if questions_count > 0 else "0.0%",
            "Out of Context": out_of_context_count,
            "Invalid Format": invalid_format_count,
            "Avg Reasoning Length (chars)": avg_reasoning_len
        }

        chart = create_pie_chart(valid_count, out_of_context_count, invalid_format_count, f"{template_name} Adherence")

        return mo.vstack([
            chart,
            mo.table([metrics])
        ])

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Improvement Approach 1: Few-Shot Demonstrations

    In this approach, we provide the model with several concrete examples of how to translate medical questions into Cypher queries. This helps the model understand the expected format, the available labels, and the target relationship (`ISA`).
    """)
    return


@app.cell
def _(mo):
    FEW_SHOT_SYSTEM_PROMPT = r"""You are a Cypher query expert for a medical knowledge graph.
    Your task is to translate natural language questions into Cypher queries.

    The graph contains entities with UMLS semantic labels.
    Key node labels include: ENTITY, CONCEPT, DISEASE_OR_SYNDROME, SIGN_OR_SYMPTOM, PHARMACOLOGIC_SUBSTANCE, GENE_OR_GENOME, etc.
    Nodes often have multiple labels. Use the most specific one if known, or ENTITY if unsure.
    All entities have a 'name' property.

    Available relationship:
    - [:ISA] : Connects an ENTITY to a CONCEPT.

    Output format:
    1. A <think> block explaining your step-by-step reasoning.
    2. The Cypher query wrapped in <cypher> tags.
    Ensure the query follows the structure: MATCH ... [WHERE ...] RETURN ... [ORDER BY ...] [LIMIT ...]

    ### Examples

    Question: What is Diabetes?
    <think>
    The question asks for information about 'Diabetes'. I will find the ENTITY with name 'Diabetes'.
    </think>
    <cypher>
    MATCH (e:ENTITY {name: 'Diabetes'}) RETURN e
    </cypher>

    Question: What category does Aspirin belong to?
    <think>
    The question asks for the category of 'Aspirin'. Categories are represented as CONCEPT nodes connected via the ISA relationship.
    </think>
    <cypher>
    MATCH (e:ENTITY {name: 'Aspirin'})-[:ISA]->(c:CONCEPT) RETURN c.name
    </cypher>

    Question: Symptoms of Hypertension.
    <think>
    The question is about 'Hypertension', which is a DISEASE_OR_SYNDROME. I will search for this entity.
    </think>
    <cypher>
    MATCH (e:DISEASE_OR_SYNDROME {name: 'Hypertension'}) RETURN e
    </cypher>"""

    FEW_SHOT_USER_PROMPT_TEMPLATE = r"""Convert the following medical question to a Cypher query.

    Question: {question}"""

    mo.md(
        f"""
    We consider the following two prompts for this improvement strategy:

    ### 1. System Prompt
    Establishes a specialized persona that understands UMLS medical labels and the specific graph relationship (`ISA`).

    ```text
    {FEW_SHOT_SYSTEM_PROMPT}
    ```

    ### 2. User Prompt Template
    Simple wrapper for the current question.

    ```text
    {FEW_SHOT_USER_PROMPT_TEMPLATE}
    ```
        """
    )
    return FEW_SHOT_SYSTEM_PROMPT, FEW_SHOT_USER_PROMPT_TEMPLATE


@app.cell
def _(
    FEW_SHOT_SYSTEM_PROMPT,
    FEW_SHOT_USER_PROMPT_TEMPLATE,
    QuestionToQueryPipeline,
    mo,
    questions_and_answers,
):
    mo.md("### Running Few-Shot Pipeline")

    # We use a subset for faster demonstration if needed, but here we use the full 100
    few_shot_request = QuestionToQueryPipeline.PipelineRunRequest(
        data=questions_and_answers,
        system_prompt=FEW_SHOT_SYSTEM_PROMPT,
        user_prompt_template=FEW_SHOT_USER_PROMPT_TEMPLATE,
        model_name="gemini-1.5-flash", 
        dataset="medquad_few_shot",
        allow_correction=False
    )

    # pipeline.run(few_shot_request)
    return


if __name__ == "__main__":
    app.run()
