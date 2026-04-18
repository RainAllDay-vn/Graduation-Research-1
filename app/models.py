from datetime import datetime
from typing import Optional, NamedTuple


class SystemPrompt(NamedTuple):
    id: str
    content: str
    created_at: datetime

class UserPromptTemplate(NamedTuple):
    id: str
    content: str
    created_at: datetime

class ModelRequest(NamedTuple):
    system_prompt: str
    user_prompt: str
    previous_answer_prompt: Optional[str] = None
    correction_prompt: Optional[str] = None
    dataset: Optional[str] = None
    question: Optional[str] = None
    type: Optional[str] = None

class CachedModelRequest(NamedTuple):
    id: int
    model_name: str
    dataset: Optional[str] = None
    question: Optional[str] = None
    type: Optional[str] = None
    system_prompt: SystemPrompt
    user_prompt_template: UserPromptTemplate
    previous_request_id: Optional[int] = None
    correction_prompt_template: Optional[UserPromptTemplate] = None
    valiation_result: Optional[str] = None
    include_reasoning: bool = True
    response: str
    reasoning: Optional[str] = None
    context_length_exceeded: bool = False
    created_at: datetime