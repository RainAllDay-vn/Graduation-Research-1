# Research Progress Checklist: LLM-KG-RAG Medical Chatbot

> [!IMPORTANT]
> This checklist tracks the implementation of **Ontology-Constrained Knowledge Distillation** for generating Cypher queries in a medical context.

## 1. Foundational Knowledge & Environment
- [x] **Knowledge Graph Foundations**: Understanding nodes, edges, triples (RDF), and graph theory basics. 
- [x] **Neo4j & Cypher**: Mastering the graph database, query syntax, and connection logic. 
- [x] **LLM Basics**: Understanding Transformer architectures and prompt engineering (GPT/Gemini). 
- [x] **Environment Setup**: Python environment, Neo4j instance, and necessary library installations (py2neo, langchain, etc.). 
- [x] **Research Roadmap**: Defining the teacher-student distillation architecture. 

## 2. Ontology & KG Engineering
- [ ] **Ontology Design**: Defining classes (`Disease`, `Symptom`, `Drug`) and relationships (`treatedBy`, `causes`). 
- [ ] **Data Ingestion**: Developing scripts to load datasets (e.g., KQA Pro) into Neo4j. 
- [ ] **Medical Ontology refinement**: Integrating standard medical ontologies (UMLS, SNOMED CT) or refining custom schemas. 
- [ ] **Constraints & Reasoning**: Implementing schema constraints in Neo4j to ensure data integrity. 

## 3. Teacher Pipeline (Gold Standard)
- [ ] **Semantic Parsing**: Using large LLMs (GPT-4/Gemini) to convert natural language to Cypher. 
- [ ] **KG Execution**: Executing generated Cypher queries on Neo4j and handling results. 
- [ ] **RAG Integration**: Implementing retrieval of medical documents/guidelines to augment the context. 
- [ ] **Answer Synthesis**: Teacher model generating the final natural language answer based on KG + RAG results. 

## 4. Distillation Dataset Construction
- [ ] **Data Collection**: Logging (Question, Cypher, Result) triplets from the Teacher pipeline. 
- [ ] **Dataset Cleaning**:
    - [ ] Normalizing Cypher syntax. 
    - [ ] Removing/Filtering erroneous or "hallucinated" queries. 
- [ ] **Ontology Validation**: Checking if Teacher-generated queries follow the domain/range constraints. 
- [ ] **Synthetic Data Generation**:
    - [ ] Generating questions from KG templates (e.g., `Disease -> treatedBy -> Drug`). 
    - [ ] Paraphrasing questions using LLMs to increase linguistic diversity. 

## 5. Student Model Development
- [ ] **Model Selection**: Choosing a small LLM (LLaMA, Mistral, or Qwen) for distillation. 
- [ ] **Input/Output Formatting**: Designing the prompt for the student (Question + Schema -> Cypher). 
- [ ] **Fine-tuning Setup**: Preparing the training loop for the student model using the distillation dataset. 
- [ ] **Inference Optimization**: Ensuring the student model can run efficiently with low latency. 

## 6. Ontology-Constrained Training (Core Research)
- [ ] **Knowledge Distillation (KD) Loss**: Implementing the core loss function to match teacher outputs. 
- [ ] **Ontology Constraint Loss**:
    - [ ] Developing the **Ontology Checker** module. 
    - [ ] Designing the penalty mechanism for schema-violating queries during training. 
- [ ] **Execution-Guided Loss (RL/Feedback)**:
    - [ ] Implementing a feedback loop that penalizes queries resulting in Neo4j syntax errors. 
    - [ ] Penalizing queries that return empty sets for valid factual questions. 

## 7. Experimental Setup & Benchmarking
- [ ] **Target Datasets**:
    - [ ] LC-QuAD 2.0 (Medical conversion). 
    - [ ] WebQSP / MetaQA integration. 
- [ ] **Evaluation Metrics**:
    - [ ] **Query Exact Match**: Comparing student queries to teacher/ground truth. 
    - [ ] **Execution Accuracy**: Checking if queries return the correct entities. 
    - [ ] **Ontology Consistency Rate**: Measuring how often the student follows the schema. 
    - [ ] **End-to-End QA Accuracy**: Final answer correctness. 
- [ ] **Baseline Definition**:
    - [ ] Prompt-only (Zero-shot) baseline. 
    - [ ] Standard Fine-tuned (No constraints) baseline. 

## 8. Result Analysis & Documentation
- [ ] **Cost & Latency Analysis**: Comparing Student efficiency vs. Teacher API calls. 
- [ ] **Result Visualization**: Charting the improvement with Ontology Constraints. 
- [ ] **Final Documentation**: Updating `goals.md`, progress reports, and writing the final thesis/paper. 
