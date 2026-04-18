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
    model_name: str
    system_prompt: str
    user_prompt: str
    previous_answer_prompt: Optional[str] = None
    correction_prompt: Optional[str] = None
    dataset: Optional[str] = None
    question: Optional[str] = None
    type: Optional[str] = None

class ModelResponse(NamedTuple):
    response: str
    reasoning: Optional[str] = None
    context_length_exceeded: bool = False

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

    def from_request_and_response(request: ModelRequest, response: ModelResponse) -> "CachedModelRequest":
        return CachedModelRequest(
            model_name=request.model_name,
            dataset=request.dataset,
            question=request.question,
            type=request.type,
            system_prompt=request.system_prompt,
            user_prompt_template=request.user_prompt_template,
            previous_request_id=request.previous_request_id,
            correction_prompt_template=request.correction_prompt_template,
            response=response.response,
            reasoning=response.reasoning,
            context_length_exceeded=response.context_length_exceeded,
            created_at=datetime.now()
        )