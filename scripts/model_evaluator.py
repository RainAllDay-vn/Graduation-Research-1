import json
import logging
import os
import re
import argparse
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union

import litellm

# Disable litellm verbose logging
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

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
        self._lock = threading.Lock()
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
            with self._lock:
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
        """Stores a response in the cache with thread locking."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO ai_cache (model_name, system_prompt, user_prompt, response)
                        VALUES (?, ?, ?, ?)
                    ''', (self.model_name, system_prompt, user_prompt, response))
                    conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to cache response: {e}")

    def call_model(self, prompt: str, system_prompt: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Calls the model using litellm and returns the response content and logprobs.
        Checks the sqlite cache first unless force_refresh is True.
        """
        if not force_refresh:
            cached = self._get_cached_response(system_prompt, prompt)
            if cached is not None:
                try:
                    # Attempt to parse as JSON for structured logs
                    data = json.loads(cached)
                    if isinstance(data, dict) and "content" in data:
                        logger.info("Using structured cached response from database.")
                        return data
                except json.JSONDecodeError:
                    pass
                
                logger.info("Using legacy cached response from database.")
                return {"content": cached, "logprobs": None}

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Check if model is Gemini (which currently rejects logprobs params)
            is_gemini = self.model_name.lower().startswith("gemini")
            
            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "api_key": self.api_key,
                "api_base": self.api_base,
                "drop_params": True
            }
            
            # Explicitly only add logprobs if NOT a Gemini model to avoid 400 errors
            if not is_gemini:
                completion_kwargs.update({
                    "logprobs": True,
                    "top_logprobs": 5
                })
            
            response = litellm.completion(**completion_kwargs)
            
            content = response.choices[0].message.content or ""
            
            # Extract logprobs if available in the response
            logprobs = None
            try:
                choice = response.choices[0]
                if hasattr(choice, 'logprobs') and choice.logprobs:
                    logprobs = getattr(choice.logprobs, 'content', None)
            except Exception as e:
                logger.debug(f"Logprobs not available for this response: {e}")

            result_data = {"content": content, "logprobs": logprobs}
            
            # Cache the result as JSON string
            self._cache_response(system_prompt, prompt, json.dumps(result_data, ensure_ascii=False))
                
            return result_data
            
        except Exception as e:
            error_msg = f"Failed to call model '{self.model_name}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_answers(
        self, 
        dataset: List[Dict[str, Any]], 
        prompt_template: str,
        max_workers: int = 5,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Evaluates the model's accuracy on a given dataset in parallel.
        """
        total_count = len(dataset)
        results = [None] * total_count # Maintain order
        
        system_prompt = (
            "You are an expert in Cypher query language. "
            "Translate the natural language question into a Cypher query. "
            "Return only the raw Cypher query, without markdown blocks or explanations."
        )

        def process_item(index: int, entry: Dict[str, Any]) -> Dict[str, Any]:
            try:
                logger.debug(f"Handling item {index+1}/{total_count} (UID: {entry.get('uid')})")
                prompt = prompt_template.format(**entry)
                model_output = self.call_model(prompt, system_prompt, force_refresh=force_refresh)
                
                answer = model_output["content"].strip()
                answer = re.sub(r'^```(cypher)?\n?', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = re.sub(r'\n?```$', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = answer.strip()
                
                return {
                    **entry, 
                    "answer": answer,
                    "logprobs": model_output.get("logprobs")
                }
            except Exception as e:
                logger.error(f"Failed to evaluate item {index+1}: {str(e)}")
                return {**entry, "answer": "", "logprobs": None}

        logger.info(f"Starting evaluation with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Schedule all tasks
            future_to_index = {
                executor.submit(process_item, i, entry): i 
                for i, entry in enumerate(dataset)
            }
            
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                completed += 1
                if completed % 5 == 0 or completed == total_count:
                    logger.info(f"Progress: {completed}/{total_count} items completed.")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on Cypher generation.")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="Model name (litellm format)")
    parser.add_argument("--api-key", type=str, help="API key for the model")
    parser.add_argument("--force", action="store_true", help="Force refresh cache and call AI")
    parser.add_argument("--db", type=str, default="cache/ai_cache.db", help="Path to sqlite cache database")
    parser.add_argument("--dataset", type=str, default="dataset/mini", help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (e.g., train, test)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers for API calls")
    
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
        results = evaluator.get_answers(
            dataset=dataset, 
            prompt_template=template, 
            max_workers=args.workers, 
            force_refresh=args.force
        )
        
        # Save results to dataset folder
        output_filename = f"results_{args.model.replace('/', '_')}_{args.split}.json"
        dataset_dir = args.dataset if os.path.isdir(args.dataset) else "dataset"
        output_file = os.path.join(dataset_dir, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation results to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
