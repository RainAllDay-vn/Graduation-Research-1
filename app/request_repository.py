import sqlite3
from datetime import datetime
from typing import Optional
from app.models import (
    ModelRequest,
    CachedModelRequest,
    SystemPrompt,
    UserPromptTemplate,
    CorrectionPromptTemplate
)

class RequestRepository:
    def __init__(self, db_path: str = "./cache/repository.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # System Prompt Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_prompts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)

            # User Prompt Template Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_prompt_templates (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)

            # Correction Prompt Template Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correction_prompt_templates (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)

            # Cached Model Request Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    dataset TEXT,
                    question TEXT,
                    type TEXT,
                    system_prompt_id TEXT NOT NULL,
                    user_prompt_template_id TEXT NOT NULL,
                    previous_request_id INTEGER,
                    correction_prompt_template_id TEXT,
                    validation_result TEXT,
                    include_reasoning INTEGER NOT NULL,
                    response TEXT NOT NULL,
                    reasoning TEXT,
                    retries INTEGER NOT NULL DEFAULT 0,
                    context_length_exceeded INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (system_prompt_id) REFERENCES system_prompts (id),
                    FOREIGN KEY (user_prompt_template_id) REFERENCES user_prompt_templates (id),
                    FOREIGN KEY (correction_prompt_template_id) REFERENCES correction_prompt_templates (id)
                )
            """)
            conn.commit()

    def save_system_prompt(self, prompt: SystemPrompt):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO system_prompts (id, content, created_at) VALUES (?, ?, ?)",
                (prompt.id, prompt.content, prompt.created_at.isoformat())
            )
            conn.commit()

    def save_user_prompt_template(self, template: UserPromptTemplate):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO user_prompt_templates 
                (id, content, created_at) VALUES (?, ?, ?)""",
                (template.id, template.content, template.created_at.isoformat())
            )
            conn.commit()

    def save_correction_prompt_template(self, template: CorrectionPromptTemplate):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO correction_prompt_templates 
                (id, content, created_at) VALUES (?, ?, ?)""",
                (template.id, template.content, template.created_at.isoformat())
            )
            conn.commit()

    def get_system_prompt(self, prompt_id: str) -> Optional[SystemPrompt]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_prompts WHERE id = ?", (prompt_id,))
            row = cursor.fetchone()
            if row:
                return SystemPrompt(
                    id=row["id"],
                    content=row["content"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
            return None

    def get_user_prompt_template(self, template_id: str) -> Optional[UserPromptTemplate]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_prompt_templates WHERE id = ?", (template_id,))
            row = cursor.fetchone()
            if row:
                return UserPromptTemplate(
                    id=row["id"],
                    content=row["content"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
            return None

    def get_correction_prompt_template(
        self,
        template_id: str
    ) -> Optional[CorrectionPromptTemplate]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM correction_prompt_templates WHERE id = ?", 
                (template_id,)
            )
            row = cursor.fetchone()
            if row:
                return CorrectionPromptTemplate(
                    id=row["id"],
                    content=row["content"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
            return None

    def save_request(self, request: CachedModelRequest) -> int:
        # Ensure prompts are saved first
        self.save_system_prompt(request.system_prompt)
        self.save_user_prompt_template(request.user_prompt_template)
        if request.correction_prompt_template:
            self.save_correction_prompt_template(request.correction_prompt_template)

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO requests (
                        model_name, dataset, question, type, system_prompt_id, 
                        user_prompt_template_id, previous_request_id, 
                        correction_prompt_template_id, validation_result, 
                        include_reasoning, response, reasoning, retries,
                        context_length_exceeded, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.model_name,
                        request.dataset,
                        request.question,
                        request.type,
                        request.system_prompt.id,
                        request.user_prompt_template.id,
                        request.previous_request_id,
                        request.correction_prompt_template.id
                            if request.correction_prompt_template else None,
                        request.validation_result,
                        1 if request.include_reasoning else 0,
                        request.response,
                        request.reasoning,
                        request.retries,
                        1 if request.context_length_exceeded else 0,
                        request.created_at.isoformat()
                    )
                )
        finally:
            conn.close()

    def _row_to_cached_model_request(self, row: sqlite3.Row) -> CachedModelRequest:
        system_prompt = self.get_system_prompt(row["system_prompt_id"])
        user_prompt_template = self.get_user_prompt_template(row["user_prompt_template_id"])
        correction_prompt_template = None
        if row["correction_prompt_template_id"]:
            correction_prompt_template = \
                self.get_correction_prompt_template(row["correction_prompt_template_id"])

        return CachedModelRequest(
            id=row["id"],
            model_name=row["model_name"],
            dataset=row["dataset"],
            question=row["question"],
            type=row["type"],
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            previous_request_id=row["previous_request_id"],
            correction_prompt_template=correction_prompt_template,
            validation_result=row["validation_result"],
            include_reasoning=bool(row["include_reasoning"]),
            response=row["response"],
            reasoning=row["reasoning"],
            retries=row["retries"],
            context_length_exceeded=bool(row["context_length_exceeded"]),
            created_at=datetime.fromisoformat(row["created_at"])
        )

    def get_request_by_id(self, request_id: int) -> Optional[CachedModelRequest]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM requests WHERE id = ?", (request_id,))
            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_cached_model_request(row)

    def get_request_by_metadata(self, request: ModelRequest) -> Optional[CachedModelRequest]:
        with self._get_connection() as conn:
            model_name=request.model_name
            dataset=request.dataset
            question=request.question
            request_type = request.type

            cursor = conn.cursor()
            cursor.execute(
                """
                    SELECT * FROM requests 
                        WHERE model_name = ? 
                        AND dataset = ?
                        AND question = ? 
                        AND type = ?
                    ORDER BY created_at
                """,
                (model_name, dataset, question, request_type)
            )

            rows = cursor.fetchall()
            results = [self._row_to_cached_model_request(row) for row in rows]
            if not results:
                return None

            for result in results:
                if result.validation_result is None:
                    return result

            return results[0]
