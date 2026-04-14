from litellm import ModelResponse
import logging
import os
import hashlib
import argparse
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Any, Optional

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
                    "model_name": "TEXT",
                    "system_prompt_id": "TEXT",
                    "question": "TEXT",
                    "include_reasoning": "INTEGER",
                    "response_text": "TEXT",
                    "reasoning_content": "TEXT",
                    "token_1": "TEXT",
                    "token_2": "TEXT",
                    "token_3": "TEXT",
                    "token_4": "TEXT",
                    "token_5": "TEXT",
                    "logprob_1": "REAL",
                    "logprob_2": "REAL",
                    "logprob_3": "REAL",
                    "logprob_4": "REAL",
                    "logprob_5": "REAL",
                    "context_length_exceeded": "INTEGER",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                },
                "pk": ["model_name", "system_prompt_id", "question", "include_reasoning"],
                "fk": "FOREIGN KEY(system_prompt_id) REFERENCES system_prompts(system_prompt_id)"
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
                
                # Safely extract top 5 logprobs if available
                lps = [None] * 5
                tokens = [None] * 5
                try:
                    if self.logprobs and hasattr(choice, "logprobs") and choice.logprobs:
                        if hasattr(choice.logprobs, "content") and choice.logprobs.content:
                            for i, lp_info in enumerate(choice.logprobs.content[:self.top_logprobs]):
                                if i < 5:
                                    lps[i] = lp_info.get("logprob") if isinstance(lp_info, dict) else getattr(lp_info, "logprob", None)
                                    tokens[i] = lp_info.get("token") if isinstance(lp_info, dict) else getattr(lp_info, "token", None)
                except Exception as e:
                    logger.debug(f"Could not extract logprobs: {e}")
                    
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO requests 
                    (model_name, system_prompt_id, question, include_reasoning, response_text, reasoning_content,
                     token_1, token_2, token_3, token_4, token_5,
                     logprob_1, logprob_2, logprob_3, logprob_4, logprob_5, context_length_exceeded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self.model_name, system_prompt_id, question, int(self.include_reasoning), response_text, reasoning_content,
                     tokens[0], tokens[1], tokens[2], tokens[3], tokens[4],
                     lps[0], lps[1], lps[2], lps[3], lps[4], context_length_exceeded)
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
                    SELECT response_text, reasoning_content, context_length_exceeded, token_1, token_2, token_3, token_4, token_5, logprob_1, logprob_2, logprob_3, logprob_4, logprob_5
                    FROM requests 
                    WHERE model_name=? AND system_prompt_id=? AND question=? AND include_reasoning=?
                    """,
                    (self.model_name, system_prompt_id, question, int(include_reasoning))
                )
                row = cursor.fetchone()
                if row:
                    logger.info("Cache hit.")
                    return dict(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch cached response: {e}")
        return {}

    def fetch_all_cached_responses(self, system_prompt: str, questions: list[str], include_reasoning: bool) -> list[dict[str, Any]]:
        """
        Fetches all cached model responses from the SQLite database.
        """
        system_prompt_id = self._get_system_prompt_id(system_prompt)
        results = []
        if not questions:
            return results
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                placeholders = ','.join(['?'] * len(questions))
                query = f"""
                    SELECT model_name, question, response_text, reasoning_content, context_length_exceeded, token_1, token_2, token_3, token_4, token_5, logprob_1, logprob_2, logprob_3, logprob_4, logprob_5
                    FROM requests 
                    WHERE system_prompt_id=? AND include_reasoning=? AND question IN ({placeholders})
                """
                params = [system_prompt_id, int(include_reasoning)] + questions
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch cached responses: {e}")
        return results

    def call_model_single(self, system_prompt: str, user_prompt: str, force_refresh: bool = False) -> dict[str, Any]:
        """
        Calls the model using litellm and returns the response content.
        Checks the sqlite cache first unless force_refresh is True.
        """
        if not force_refresh:
            cached = self._fetch_cached_response(system_prompt, user_prompt, self.include_reasoning)
            if cached:
                return cached
                
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
            self._cache_response(system_prompt, user_prompt, response)
            return self._fetch_cached_response(system_prompt, user_prompt, self.include_reasoning) or {}
            
        except Exception as e:
            logger.error(f"LiteLLM call failed: {e}")
            return {}

    def call_model(self, input_data: list[tuple[str, list[str]]], force_refresh: bool = False) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Evaluates the model's accuracy on a given set of questions.
        """

        logger.info(f"Starting evaluation of {sum(len(q) for _, q in input_data)} questions using {self.model_name}.")
        
        def process_question(sp, q):
            return (sp, q), self.call_model_single(sp, q, force_refresh=force_refresh)

        results = {}
        # ThreadPoolExecutor to run API calls concurrently
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for system_prompt, user_prompts in input_data:
                for user_prompt in user_prompts:
                    futures.append(executor.submit(process_question, system_prompt, user_prompt))
            
            for future in as_completed(futures):
                try:
                    (sp, q), result = future.result()
                    results[(sp, q)] = result
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
