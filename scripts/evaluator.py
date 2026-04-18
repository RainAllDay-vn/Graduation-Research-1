import os
import pandas as pd
from scripts.ontology_checker import OntologyChecker
from knowledge_graph import KnowledgeGraph
from model_provider import ModelProvider, CallModelRequest

class Evaluator:
    def evaluate(
        self, 
        model_name: str, 
        dataset_path: str, 
        system_prompt: str,
        reasoning: bool = True,
        allow_self_correction: bool = False,
        limit: int = None
    ):
        model_provider = ModelProvider(model_name, include_reasoning=reasoning)
        knowledge_graph = KnowledgeGraph()
        if allow_self_correction:
            checker = OntologyChecker(knowledge_graph)
            checker_function = checker.check_validity
        else:
            checker_function = None
        
        dataset_path = os.path.join(dataset_path, "test.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        df = pd.read_csv(dataset_path, usecols=["question"])
        if limit:
            df = df.head(limit)
        input_data = [
            CallModelRequest(
                system_prompt=system_prompt, 
                user_prompt=question,
                dataset=os.path.basename(dataset_path),
                question=question,
                type="EVALUATION"
            ) for question in df['question']
        ]
        result = model_provider.call_model(input_data, checker_function=checker_function)

        print(len(result))