from litellm import ModelResponse
import logging
import os
import hashlib
import argparse
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Any, Optional, Callable

import litellm

litellm.request_timeout = None  # No timeout for requests

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
    def __init__(
        self, 
        model_name: str, 
        api_key: Optional[str] = None, 
        api_base: Optional[str] = None, 
        provider: Optional[str] = None, 
        db_path: Optional[str] = None, 
        workers: Optional[int] = None,
        logprobs: bool = False, 
        top_logprobs: int = 5, 
        include_reasoning: bool = True
    ):

        """
        Initializes the ModelEvaluator with model details and a database path.
        """
        load_dotenv()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY")
        self.api_base = api_base or os.environ.get("LITELLM_API_URL")
        self.provider = provider or os.environ.get("LITELLM_PROVIDER")
        self.db_path = db_path or os.environ.get("LITELLM_CACHE_PATH", "cache/cache.db")
        
        # Resolve workers from arg, then env, then default
        env_workers = os.environ.get("LITELLM_WORKER")
        self.workers = workers if workers is not None else (int(env_workers) if env_workers else 8)

        self.logprobs = logprobs
        self.top_logprobs = min(top_logprobs, 5)
        self.include_reasoning = include_reasoning
        self._init_db()
        
        logger.info(
            f"ModelEvaluator initialized with Configuration:\n"
            f"  Model Name:         {self.model_name}\n"
            f"  API Key:            {'HIDDEN' if self.api_key else 'None'}\n"
            f"  API Base:           {self.api_base}\n"
            f"  Provider:           {self.provider}\n"
            f"  DB Path:            {self.db_path}\n"
            f"  Workers:            {self.workers}\n"
            f"  Logprobs:           {self.logprobs} (Top {self.top_logprobs})\n"
            f"  Include Reasoning:  {self.include_reasoning}"
        )

    def _init_db(self):
        """Initializes the SQLite database folder."""
        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            logger.info(f"Creating directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)

        tables = {
            "system_prompts": {
                "columns": {
                    "system_prompt_id": "TEXT PRIMARY KEY",
                    "content": "TEXT",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                },
                "pk": None # Defined inline above
            },
            "requests": {
                "columns": {
                    "request_id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "model_name": "TEXT",
                    "system_prompt_id": "TEXT",
                    "user_prompt": "TEXT",
                    "include_reasoning": "INTEGER",
                    "response_text": "TEXT",
                    "reasoning_content": "TEXT",
                    "context_length_exceeded": "INTEGER",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                },
                "pk": None, # Defined inline above
                "fk": "FOREIGN KEY(system_prompt_id) REFERENCES system_prompts(system_prompt_id)",
                "unique": ["model_name", "system_prompt_id", "user_prompt", "include_reasoning"]
            },
            "token_logprobs": {
                "columns": {
                    "logprob_id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "request_id": "INTEGER",
                    "token_position": "INTEGER",
                    "token_text": "TEXT",
                    "logprob": "REAL"
                },
                "pk": None,
                "fk": "FOREIGN KEY(request_id) REFERENCES requests(request_id) ON DELETE CASCADE"
            }
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable Foreign Key support
                conn.execute("PRAGMA foreign_keys = ON;")
                cursor = conn.cursor()

                for table_name, config in tables.items():
                    # Check if table exists
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    if not cursor.fetchone():
                        # Build CREATE statement
                        col_defs = [f"{name} {dtype}" for name, dtype in config["columns"].items()]
                        
                        constraints = []
                        if config.get("pk"):
                            constraints.append(f"PRIMARY KEY ({', '.join(config['pk'])})")
                        if config.get("fk"):
                            constraints.append(config["fk"])
                        if config.get("unique"):
                            constraints.append(f"UNIQUE ({', '.join(config['unique'])})")
                        
                        query = f"CREATE TABLE {table_name} ({', '.join(col_defs + constraints)})"
                        cursor.execute(query)
                    else:
                        # Safety Check: Verify existing schema
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        existing_cols = {row[1] for row in cursor.fetchall()}
                        required_cols = set(config["columns"].keys())
                        
                        missing = required_cols - existing_cols
                        if missing:
                            logger.info(f"Migration: Adding missing columns {missing} to table {table_name}")
                            for col in missing:
                                dtype = config["columns"][col]
                                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype}")

        except sqlite3.Error as e:
            raise RuntimeError(f"Database error during initialization: {e}")


    def _get_system_prompt_id(self, system_prompt: str) -> str:
        """Returns a deterministic hash ID for the system prompt."""
        return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()

    def _cache_response(self, system_prompt: str, question: str, response: ModelResponse):
        """Stores a response in the cache."""
        system_prompt_id = self._get_system_prompt_id(system_prompt)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO system_prompts (system_prompt_id, content) VALUES (?, ?)",
                    (system_prompt_id, system_prompt)
                )
                
                choice = response.choices[0]
                response_text = choice.message.content
                reasoning_content = getattr(choice.message, "reasoning_content", None)
                finish_reason = getattr(choice, "finish_reason", "")
                context_length_exceeded = 1 if finish_reason == "length" else 0
                
                # Insert or Replace in requests table
                # Note: We use INSERT OR REPLACE which triggers DELETE on UNIQUE conflict if ON CONFLICT clause is not specified?
                # Actually, SQLite's INSERT OR REPLACE will delete the old row, and since we have ON DELETE CASCADE, it will delete old logprobs.
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO requests 
                    (model_name, system_prompt_id, user_prompt, include_reasoning, response_text, reasoning_content, context_length_exceeded)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self.model_name, system_prompt_id, question, int(self.include_reasoning), response_text, reasoning_content, context_length_exceeded)
                )
                
                request_id = cursor.lastrowid
                
                # Extract and insert logprobs if available
                if self.logprobs and hasattr(choice, "logprobs") and choice.logprobs:
                    if hasattr(choice.logprobs, "content") and choice.logprobs.content:
                        logprob_inserts = []
                        for i, lp_info in enumerate(choice.logprobs.content):
                            # Respect top_logprobs limit if needed, though usually we want all tokens' main logprob
                            # If we want ALL tokens, we just iterate.
                            token = lp_info.get("token") if isinstance(lp_info, dict) else getattr(lp_info, "token", None)
                            lprob = lp_info.get("logprob") if isinstance(lp_info, dict) else getattr(lp_info, "logprob", None)
                            logprob_inserts.append((request_id, i, token, lprob))
                        
                        if logprob_inserts:
                            cursor.executemany(
                                "INSERT INTO token_logprobs (request_id, token_position, token_text, logprob) VALUES (?, ?, ?, ?)",
                                logprob_inserts
                            )

                conn.commit()
                logger.info(f"Successfully cached response for question: {question[:30]}...")
        except sqlite3.Error as e:
            logger.error(f"Failed to cache response: {e}")

    def _fetch_cached_response(self, system_prompt: str, question: str, include_reasoning: bool) -> dict[str, Any]:
        """
        Fetches cached model response from the SQLite database.
        """
        system_prompt_id = self._get_system_prompt_id(system_prompt)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT request_id, response_text, reasoning_content, context_length_exceeded
                    FROM requests 
                    WHERE model_name=? AND system_prompt_id=? AND user_prompt=? AND include_reasoning=?
                    """,
                    (self.model_name, system_prompt_id, question, int(include_reasoning))
                )
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    # Fetch associated logprobs
                    cursor.execute(
                        "SELECT token_text, logprob FROM token_logprobs WHERE request_id=? ORDER BY token_position",
                        (result["request_id"],)
                    )
                    lps_rows = cursor.fetchall()
                    result["logprobs"] = [dict(lp) for lp in lps_rows]
                    logger.info("Cache hit.")
                    return result
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch cached response: {e}")
        return {}

    def call_model_single(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Calls the model using litellm and returns the response content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.top_logprobs
        if self.include_reasoning:
            # LiteLLM's standard reasoning trigger
            kwargs["include_reasoning"] = True
            # SGLang specific trigger for Qwen/DeepSeek
            kwargs["extra_body"] = {"enable_thinking": True}
            
        model_id = self.model_name
        if self.provider and not model_id.startswith(f"{self.provider}/"):
            model_id = f"{self.provider}/{self.model_name}"

        try:
            logger.info(f"Calling model {self.model_name}...")
            response = litellm.completion(
                model=model_id,
                messages=messages,
                api_key=self.api_key,
                **kwargs
            )
            return response
            
        except Exception as e:
            logger.error(f"LiteLLM call failed: {e}")
            return {}

    def call_model_with_checker(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        checker: Callable[[str], str], 
        max_retries: int = 3, 
    ) -> dict[str, Any]:
        """
        Calls the model and validates the response using the provided checker.
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling model {self.model_name} (Attempt {attempt + 1}/{max_retries})...")
                response: dict[str, Any] = self.call_model_single(system_prompt, user_prompt)
                
                response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Check validity
                check_result = checker(response_text)
                if check_result == "OK":
                    return response
                
                # If not valid, prepare for retry with feedback
                logger.warning(f"Model response failed validation (Attempt {attempt + 1}): {check_result}")
                    
            except Exception as e:
                logger.error(f"LiteLLM call failed on attempt {attempt + 1}: {e}")
                
        logger.error(f"Failed to get a valid response after {max_retries} attempts.")
        return {}

    def call_model(self, input_data: list[tuple[str, str]], checker: Optional[Callable[[str], str]] = None, force_refresh: bool = False) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Evaluates the model's accuracy on a given set of questions.
        """
        logger.info(f"Starting evaluation of {len(input_data)} questions using {self.model_name}.")
        
        def process_question(system_prompt: str, user_prompt: str, checker: Optional[Callable[[str], str]] = None):
            if not force_refresh:
                cached_response = self._fetch_cached_response(system_prompt, user_prompt)
                if cached_response:
                    return (system_prompt, user_prompt), cached_response
            if checker:
                response = self.call_model_with_checker(system_prompt, user_prompt, checker)
            else:
                response = self.call_model_single(system_prompt, user_prompt)
            self._cache_response(system_prompt, user_prompt, response)
            return (system_prompt, user_prompt), response

        results = {}
        # Try the first request to fail fast if there's an issue (e.g., connection error)
        try:
            system_prompt, user_prompt = input_data[0]
            logger.info("Trying first request to verify connection...")
            _, first_res = process_question(system_prompt, user_prompt, checker)
            results[(system_prompt, user_prompt)] = first_res
            
            if not first_res:
                logger.error("First request failed (returned empty). Aborting remaining evaluation.")
                return results
                
        except Exception as e:
             logger.error(f"Error processing first request, aborting: {e}")
             return results

        if len(input_data) > 1:
            # ThreadPoolExecutor to run API calls concurrently
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for system_prompt, user_prompt in input_data[1:]:
                    futures.append(executor.submit(process_question, system_prompt, user_prompt, checker))
                
                for future in as_completed(futures):
                    try:
                        (system_prompt, user_prompt), result = future.result()
                        results[(system_prompt, user_prompt)] = result
                    except Exception as e:
                        logger.error(f"Error processing question: {e}")

        return results
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on Cypher generation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B", help="Model name (litellm format)")
    parser.add_argument("--api-key", type=str, help="API key for the model")
    parser.add_argument("--force", action="store_true", help="Force refresh cache and call AI")
    parser.add_argument("--db", type=str, help="Path to sqlite cache database (Default from LITELLM_CACHE_PATH or cache/cache.db)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (Default from LITELLM_WORKER or 8)")
    parser.add_argument("--api-base", type=str, help="Base URL for the model API (Default from LITELLM_API_URL)")
    parser.add_argument("--provider", type=str, help="Provider format for the model (e.g., openai, anthropic)")
    parser.add_argument("--logprobs", action="store_true", help="Request log probabilities from the model")
    parser.add_argument("--top-logprobs", type=int, default=5, help="Number of top log probabilities to return (if logprobs enriched)")
    parser.add_argument("--thinking", action="store_true", default=True, help="Enable thinking/reasoning for models that support it")
    parser.add_argument("--no-thinking", action="store_false", dest="thinking", help="Disable thinking/reasoning for models that support it")

    args = parser.parse_args()
    
    # Mock execution for demonstration using dummy queries
    # Key resolution is handled inside ModelEvaluator.__init__
    api_key = args.api_key
    
    evaluator = ModelEvaluator(
        model_name=args.model,
        api_key=api_key,
        api_base=args.api_base,
        provider=args.provider,
        db_path=args.db,
        workers=args.workers,
        logprobs=args.logprobs,
        top_logprobs=args.top_logprobs,
        include_reasoning=args.thinking
    )
    
    sample_system_prompt = "You are a helpful AI assistant. Answer accurately."
    sample_questions = [
        "What is the capital of France?",
        "Write a Python script to reverse a string."
    ]
    
    evaluator.call_model(input_data=[(sample_system_prompt, sample_questions)], force_refresh=args.force)
