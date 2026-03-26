import json
import logging
import re
from typing import List, Dict, Any, Optional, Union

import litellm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A class for evaluating model accuracy
    """
    def __init__(self, model_name: str, api_key: str, api_base: Optional[str] = None):
        """
        Initializes the ModelEvaluator with model details.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

    def call_model(self, prompt: str, system_prompt: str) -> str:
        """
        Calls the model using litellm and returns the response content.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                api_key=self.api_key,
                api_base=self.api_base
            )
            
            content = response.choices[0].message.content
            return content if content is not None else ""
            
        except Exception as e:
            error_msg = f"Failed to call model '{self.model_name}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_answers(
        self, 
        dataset: List[Dict[str, Any]], 
        prompt_template: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluates the model's accuracy on a given dataset.
        Each entry in the returned list will have an extra 'answer' field with the model response.
        """
        results = []
        total_count = len(dataset)
        
        system_prompt = (
            "You are an expert in Cypher query language. "
            "Translate the natural language question into a Cypher query. "
            "Return only the raw Cypher query, without markdown blocks or explanations."
        )

        for i, entry in enumerate(dataset, 1):
            try:
                logger.info(f"Processing item {i}/{total_count} (UID: {entry.get('uid')})")
                prompt = prompt_template.format(**entry)
                raw_response = self.call_model(prompt, system_prompt)
                answer = raw_response.strip()
                answer = re.sub(r'^```(cypher)?\n?', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = re.sub(r'\n?```$', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = answer.strip()
                
                # Add answer to the entry
                entry_with_answer = {**entry, "answer": answer}
                results.append(entry_with_answer)
                
            except Exception as e:
                logger.error(f"Failed to evaluate item {i}: {str(e)}")
                # Append with empty answer in case of failure
                results.append({**entry, "answer": ""})
        
        return results

if __name__ == "__main__":
    # Example usage for testing (replace with your actual credentials)
    evaluator = ModelEvaluator(
        model_name="gemini/gemini-1.5-flash",
        api_key="your_api_key_here"
    )
    
    sample_dataset = [
        {
            "uid": 1, 
            "question": "What is the capital of France?", 
            "cypher_query": "MATCH (c:City {name: 'Paris'}) RETURN c"
        }
    ]
    template = "Translate this question to Cypher: {question}"
    
    try:
        results = evaluator.get_answers(sample_dataset, template)
        print(f"Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
