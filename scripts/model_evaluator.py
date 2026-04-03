import json
import logging
import os
import re
import argparse
import sqlite3
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, cast

import litellm

# Disable litellm verbose logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A class for evaluating model accuracy with SQLite-based caching.
    The cache key is (model_name, dataset_name, system_prompt_id, template_id, question, include_reasoning).
    """
    def __init__(self, model_name: str, api_key: str, api_base: Optional[str] = None, provider: Optional[str] = None, db_path: str = "cache/ai_cache.db", logprobs: bool = False, top_logprobs: int = 5, include_reasoning: bool = True):

        """
        Initializes the ModelEvaluator with model details and a database path.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.provider = provider
        self.db_path = db_path
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.include_reasoning = include_reasoning
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and ensures it's ready for use with the expected normalized schema."""
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                logger.info(f"Creating directory: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Basic check: verify required tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('ai_cache', 'system_prompts', 'user_prompt_templates')")
                tables = [row[0] for row in cursor.fetchall()]
                
                required = {'ai_cache', 'system_prompts', 'user_prompt_templates'}
                if not required.issubset(set(tables)):
                    missing = required - set(tables)
                    logger.error(f"Missing required database tables: {missing}. Please run 'scripts/migrate_cache.py' first.")
                else:
                    logger.debug("Database schema verified.")
                
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database connection: {e}")

    def _get_or_create_id(self, cursor, table, content):
        """Helper to get or create a record in a normalization table."""
        cursor.execute(f"SELECT id FROM {table} WHERE content = ?", (content,))
        result = cursor.fetchone()
        if result:
            return result[0]
        cursor.execute(f"INSERT INTO {table} (content) VALUES (?)", (content,))
        return cursor.lastrowid

    def _cache_response(self, dataset_name: str, system_prompt: str, template: str, question: str, response: str, logprobs: Optional[List[Any]] = None):
        """Stores a response in the cache with thread locking and normalization."""
        try:
            logprobs_json = json.dumps(logprobs, ensure_ascii=False) if logprobs else None
            
            # Remove redundant logprobs from response JSON if present
            try:
                resp_data = json.loads(response)
                if isinstance(resp_data, dict) and "logprobs" in resp_data:
                    del resp_data["logprobs"]
                    response = json.dumps(resp_data, ensure_ascii=False)
            except:
                pass

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                sys_id = self._get_or_create_id(cursor, "system_prompts", system_prompt)
                tmpl_id = self._get_or_create_id(cursor, "user_prompt_templates", template)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO ai_cache 
                    (model_name, dataset_name, system_prompt_id, template_id, question, include_reasoning, response, logprobs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (self.model_name, dataset_name, sys_id, tmpl_id, question, int(self.include_reasoning), response, logprobs_json))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to cache response: {e}")

    def call_model(self, question: str, prompt_template: str, system_prompt: str, dataset_name: str = "unknown", force_refresh: bool = False) -> Dict[str, Any]:
        """
        Calls the model using litellm and returns the response content and logprobs.
        Checks the sqlite cache first unless force_refresh is True.
        """
        # Prepare template with requested {{question}} format for DB
        db_template = prompt_template.replace("{question}", "{{question}}")
        prompt = prompt_template.format(question=question)


        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "api_key": self.api_key,
                "api_base": self.api_base,
                "custom_llm_provider": self.provider,
                "drop_params": True,
                "include_reasoning": self.include_reasoning
            }

            # Add SGLang-specific reasoning control for Qwen models
            if not self.include_reasoning:
                completion_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            else:
                # For consistency toggle it on if requested
                completion_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": True
                    }
                }

            if self.logprobs:
                completion_kwargs.update({
                    "logprobs": True,
                    "top_logprobs": self.top_logprobs
                })
            
            response = litellm.completion(**completion_kwargs)
            response = cast(litellm.ModelResponse, response)
            
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "length":
                logger.warning(f"Model '{self.model_name}' reached token or context limit (finish_reason: length). Content might be truncated.")

            # Extract logprobs if available in the response
            logprobs = None
            try:
                choice = response.choices[0]
                if hasattr(choice, 'logprobs') and choice.logprobs:
                    raw_logprobs = getattr(choice.logprobs, 'content', None)
                    if raw_logprobs:
                        logprobs = [lp.model_dump() if hasattr(lp, 'model_dump') else lp for lp in raw_logprobs]
            except Exception as e:
                logger.debug(f"Logprobs not available for this response: {e}")

            result_data = {"content": content, "logprobs": logprobs, "finish_reason": finish_reason}
            
            # Cache the result as JSON string
            self._cache_response(dataset_name, system_prompt, db_template, question, json.dumps(result_data, ensure_ascii=False), logprobs=logprobs)
                
            return result_data
            
        except Exception as e:
            error_msg = f"Failed to call model '{self.model_name}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def fetch_cached_responses(
        self, 
        model_name: str, 
        dataset_name: str, 
        include_reasoning: int,
        system_prompt_id: Optional[int] = None,
        template_id: Optional[int] = None,
        questions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetches cached model responses from the SQLite database.
        Returns a pandas DataFrame containing the raw results.
        """
        query = """
        SELECT 
            *,
            json_extract(response, '$.content') AS content
        FROM ai_cache 
        WHERE model_name = ? 
          AND dataset_name = ? 
          AND include_reasoning = ?
          AND json_valid(response)
        """
        params = [model_name, dataset_name, include_reasoning]

        if system_prompt_id is not None:
            query += " AND system_prompt_id = ?"
            params.append(system_prompt_id)

        if template_id is not None:
            query += " AND template_id = ?"
            params.append(template_id)

        if questions:
            placeholders = ", ".join(["?"] * len(questions))
            query += f" AND question IN ({placeholders})"
            params.extend(questions)

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_answers(
        self, 
        dataset: List[Dict[str, Any]], 
        prompt_template: str,
        dataset_name: str = "unknown",
        max_workers: int = 5,
        force_refresh: bool = False
    ) -> None:
        """
        Evaluates the model's accuracy on a given dataset in parallel.
        Filters out already cached items if not force_refresh.
        """
        system_prompt = (
            "You are an expert in Cypher query language. "
            "Translate the natural language question into a Cypher query. "
            "Return only the raw Cypher query, without markdown blocks or explanations."
        )
        db_template = prompt_template.replace("{question}", "{{question}}")

        filtered_dataset = dataset
        if not force_refresh:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get IDs
                    cursor.execute("SELECT id FROM system_prompts WHERE content = ?", (system_prompt,))
                    sys_res = cursor.fetchone()
                    cursor.execute("SELECT id FROM user_prompt_templates WHERE content = ?", (db_template,))
                    tmpl_res = cursor.fetchone()
                    
                    if sys_res and tmpl_res:
                        cursor.execute('''
                            SELECT question FROM ai_cache 
                            WHERE model_name = ? AND dataset_name = ? AND system_prompt_id = ? AND template_id = ? AND include_reasoning = ?
                        ''', (self.model_name, dataset_name, sys_res[0], tmpl_res[0], int(self.include_reasoning)))
                        cached_questions = {row[0] for row in cursor.fetchall()}
                        
                        filtered_dataset = [
                            entry for entry in dataset 
                            if entry.get("question", "") not in cached_questions
                        ]
                        
                        skipped = len(dataset) - len(filtered_dataset)
                        if skipped > 0:
                            logger.info(f"Skipping {skipped} already cached items.")
            except sqlite3.Error as e:
                logger.warning(f"Failed to check cache batch: {e}")

        total_to_process = len(filtered_dataset)
        if total_to_process == 0:
            logger.info("All items are already cached. Nothing to process.")
            return

        def process_item(index: int, entry: Dict[str, Any]) -> None:
            try:
                logger.debug(f"Handling item {index+1}/{total_to_process} (UID: {entry.get('uid')})")
                question = entry.get("question", entry.get("utterance", ""))
                self.call_model(question, prompt_template, system_prompt, dataset_name=dataset_name, force_refresh=force_refresh)
            except Exception as e:
                logger.error(f"Failed to evaluate item {index+1}: {str(e)}")

        logger.info(f"Starting evaluation of {total_to_process} items with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_item, i, entry) 
                for i, entry in enumerate(filtered_dataset)
            ]
            
            completed = 0
            for _ in as_completed(futures):
                completed += 1
                if completed % 5 == 0 or completed == total_to_process:
                    logger.info(f"Progress: {completed}/{total_to_process} items completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on Cypher generation.")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="Model name (litellm format)")
    parser.add_argument("--api-key", type=str, help="API key for the model")
    parser.add_argument("--force", action="store_true", help="Force refresh cache and call AI")
    parser.add_argument("--db", type=str, default="cache/ai_cache.db", help="Path to sqlite cache database")
    parser.add_argument("--dataset", type=str, default="dataset/mini", help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (e.g., train, test)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers for API calls")
    parser.add_argument("--api-base", type=str, help="Base URL for the model API")
    parser.add_argument("--provider", type=str, help="Provider format for the model (e.g., openai, anthropic)")
    parser.add_argument("--logprobs", action="store_true", help="Request log probabilities from the model")
    parser.add_argument("--top-logprobs", type=int, default=5, help="Number of top log probabilities to return (if logprobs enriched)")
    parser.add_argument("--thinking", action="store_true", default=True, help="Enable thinking/reasoning for models that support it")
    parser.add_argument("--no-thinking", action="store_false", dest="thinking", help="Disable thinking/reasoning for models that support it")

    
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
        api_base=args.api_base,
        provider=args.provider,
        db_path=args.db,
        logprobs=args.logprobs,
        top_logprobs=args.top_logprobs,
        include_reasoning=args.thinking
    )
    evaluator.init_db()

    template = "Translate this question into a Cypher query:\nQuestion: {question}"
    
    # Extract dataset name from parent directory of the dataset json
    dataset_name = os.path.basename(args.dataset) if args.dataset else "unknown"

    try:
        evaluator.get_answers(
            dataset=dataset, 
            prompt_template=template, 
            dataset_name=dataset_name,
            max_workers=args.workers, 
            force_refresh=args.force
        )
        
        # Final logging
        logger.info("Evaluation process complete.")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
