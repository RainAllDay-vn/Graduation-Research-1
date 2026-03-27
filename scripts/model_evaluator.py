import json
import logging
import os
import re
import argparse
import sqlite3
from typing import List, Dict, Any, Optional, Union

import litellm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A class for evaluating model accuracy with SQLite-based caching.
    The cache key is a triple (model_name, system_prompt, user_prompt).
    """
    def __init__(self, model_name: str, api_key: str, api_base: Optional[str] = None, db_path: str = "ai_cache.db"):
        """
        Initializes the ModelEvaluator with model details and a database path.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and creates the cache table."""
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                logger.info(f"Creating directory: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_cache (
                        model_name TEXT,
                        system_prompt TEXT,
                        user_prompt TEXT,
                        response TEXT,
                        PRIMARY KEY (model_name, system_prompt, user_prompt)
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")

    def _get_cached_response(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Retrieves a cached response if available."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT response FROM ai_cache 
                    WHERE model_name = ? AND system_prompt = ? AND user_prompt = ?
                ''', (self.model_name, system_prompt, user_prompt))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            logger.warning(f"Database query failed: {e}")
            return None

    def _cache_response(self, system_prompt: str, user_prompt: str, response: str):
        """Stores a response in the cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO ai_cache (model_name, system_prompt, user_prompt, response)
                    VALUES (?, ?, ?, ?)
                ''', (self.model_name, system_prompt, user_prompt, response))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to cache response: {e}")

    def call_model(self, prompt: str, system_prompt: str, force_refresh: bool = False) -> str:
        """
        Calls the model using litellm and returns the response content.
        Checks the sqlite cache first unless force_refresh is True.
        """
        if not force_refresh:
            cached = self._get_cached_response(system_prompt, prompt)
            if cached is not None:
                logger.info("Using cached response from database.")
                return cached

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
            result = content if content is not None else ""
            
            # Cache the result
            if result:
                self._cache_response(system_prompt, prompt, result)
                
            return result
            
        except Exception as e:
            error_msg = f"Failed to call model '{self.model_name}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_answers(
        self, 
        dataset: List[Dict[str, Any]], 
        prompt_template: str,
        force_refresh: bool = False
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
                raw_response = self.call_model(prompt, system_prompt, force_refresh=force_refresh)
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
    parser = argparse.ArgumentParser(description="Evaluate model on Cypher generation.")
    parser.add_argument("--model", type=str, default="gemini/gemini-1.5-flash", help="Model name (litellm format)")
    parser.add_argument("--api-key", type=str, help="API key for the model")
    parser.add_argument("--force", action="store_true", help="Force refresh cache and call AI")
    parser.add_argument("--db", type=str, default="cache/ai_cache.db", help="Path to sqlite cache database")
    parser.add_argument("--dataset", type=str, default="dataset/mini", help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (e.g., train, test)")
    
    args = parser.parse_args()

    # Load dataset
    dataset_path = os.path.join(args.dataset, f"{args.split}.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        exit(1)
        
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} items from {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        exit(1)

    evaluator = ModelEvaluator(
        model_name=args.model,
        api_key=args.api_key or "your_api_key_here",
        db_path=args.db
    )
    
    template = "Translate this question into a Cypher query:\nQuestion: {question}"
    
    try:
        results = evaluator.get_answers(dataset, template, force_refresh=args.force)
        
        # Save results for inspection
        output_file = f"results_{args.model.replace('/', '_')}_{args.split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation results to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
