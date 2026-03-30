import json
import logging
import os
import re
import argparse
import sqlite3
import threading
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
    def __init__(self, model_name: str, api_key: str, api_base: Optional[str] = None, provider: Optional[str] = None, db_path: str = "ai_cache.db", logprobs: bool = False, top_logprobs: int = 5, include_reasoning: bool = True):

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
        self._lock = threading.Lock()
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

    def _get_cached_response(self, dataset_name: str, system_prompt: str, template: str, question: str) -> Optional[str]:
        """Retrieves a cached response if available using normalized lookups."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get IDs
                    cursor.execute("SELECT id FROM system_prompts WHERE content = ?", (system_prompt,))
                    sys_res = cursor.fetchone()
                    if not sys_res: return None
                    
                    cursor.execute("SELECT id FROM user_prompt_templates WHERE content = ?", (template,))
                    tmpl_res = cursor.fetchone()
                    if not tmpl_res: return None
                    
                    cursor.execute('''
                        SELECT response, logprobs FROM ai_cache 
                        WHERE model_name = ? AND dataset_name = ? AND system_prompt_id = ? AND template_id = ? AND question = ? AND include_reasoning = ?
                    ''', (self.model_name, dataset_name, sys_res[0], tmpl_res[0], question, int(self.include_reasoning)))
                    
                    result = cursor.fetchone()
                    if result:
                        response_text, logprobs_text = result
                        try:
                            # If logprobs_text exists, it might be more up-to-date than what's inside response_text
                            data = json.loads(response_text)
                            if isinstance(data, dict):
                                if logprobs_text and ("logprobs" not in data or data["logprobs"] is None):
                                    data["logprobs"] = json.loads(logprobs_text)
                                return json.dumps(data)
                            return response_text
                        except:
                            return response_text
                    return None
        except sqlite3.Error as e:
            logger.warning(f"Database query failed: {e}")
            return None

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

            with self._lock:
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

        if not force_refresh:
            cached = self._get_cached_response(dataset_name, system_prompt, db_template, question)
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
                return {"content": cached, "logprobs": None, "finish_reason": None}

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

    def get_answers(
        self, 
        dataset: List[Dict[str, Any]], 
        prompt_template: str,
        dataset_name: str = "unknown",
        max_workers: int = 5,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Evaluates the model's accuracy on a given dataset in parallel.
        """
        total_count = len(dataset)
        results: list = [None] * total_count # Maintain order
        
        system_prompt = (
            "You are an expert in Cypher query language. "
            "Translate the natural language question into a Cypher query. "
            "Return only the raw Cypher query, without markdown blocks or explanations."
        )

        def process_item(index: int, entry: Dict[str, Any]) -> Dict[str, Any]:
            try:
                logger.debug(f"Handling item {index+1}/{total_count} (UID: {entry.get('uid')})")
                question = entry.get("question", entry.get("utterance", ""))
                model_output = self.call_model(question, prompt_template, system_prompt, dataset_name=dataset_name, force_refresh=force_refresh)
                
                content = model_output["content"]
                thinking = ""
                answer = content
                
                if "</think>" in content:
                    parts = content.split("</think>", 1)
                    thinking = parts[0].replace("<think>", "").strip()
                    answer = parts[1].strip()
                
                answer = re.sub(r'^```(cypher)?\n?', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = re.sub(r'\n?```$', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = answer.strip()
                
                return {
                    **entry, 
                    "thinking": thinking,
                    "answer": answer,
                    "logprobs": model_output.get("logprobs"),
                    "finish_reason": model_output.get("finish_reason")
                }
            except Exception as e:
                logger.error(f"Failed to evaluate item {index+1}: {str(e)}")
                return {**entry, "thinking": "", "answer": "", "logprobs": None, "finish_reason": "error"}

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

    template = "Translate this question into a Cypher query:\nQuestion: {question}"
    
    # Extract dataset name from parent directory of the dataset json
    dataset_name = os.path.basename(args.dataset) if args.dataset else "unknown"

    try:
        results = evaluator.get_answers(
            dataset=dataset, 
            prompt_template=template, 
            dataset_name=dataset_name,
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
