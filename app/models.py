from datetime import datetime
from typing import Optional, NamedTuple
from dataclasses import dataclass

class SystemPrompt(NamedTuple):
    id: str
    content: str
    created_at: datetime

class UserPromptTemplate(NamedTuple):
    id: str
    content: str
    created_at: datetime

class CorrectionPromptTemplate(NamedTuple):
    id: str
    content: str
    created_at: datetime

@dataclass
class ModelRequest:
    system_prompt: SystemPrompt
    user_prompt_template: UserPromptTemplate
    model_name: Optional[str] = None
    previous_answer_prompt: Optional[str] = None
    previous_request_id: Optional[int] = None
    correction_prompt_template: Optional[CorrectionPromptTemplate] = None
    previous_validation_result: Optional[str] = None
    dataset: Optional[str] = None
    question: Optional[str] = None
    type: Optional[str] = None
    include_reasoning: bool = True

@dataclass
class ModelResponse:
    model_name: str
    response: str
    reasoning: Optional[str] = None
    context_length_exceeded: bool = False

@dataclass
class CachedModelRequest:
    model_name: str
    system_prompt: SystemPrompt
    user_prompt_template: UserPromptTemplate
    response: str
    retries: int
    created_at: datetime
    id: Optional[int] = None
    dataset: Optional[str] = None
    question: Optional[str] = None
    type: Optional[str] = None
    previous_request_id: Optional[int] = None
    correction_prompt_template: Optional[CorrectionPromptTemplate] = None
    validation_result: Optional[str] = None
    include_reasoning: bool = True
    reasoning: Optional[str] = None
    context_length_exceeded: bool = False

    @classmethod
    def from_request_and_response(
        cls,
        request: ModelRequest,
        response: ModelResponse,
        retries: Optional[int] = 0,
        validation_result: Optional[str] = None
    ) -> "CachedModelRequest":
        return cls(
            model_name=response.model_name,
            dataset=request.dataset,
            question=request.question,
            type=request.type,
            system_prompt=request.system_prompt,
            user_prompt_template=request.user_prompt_template,
            previous_request_id=request.previous_request_id,
            correction_prompt_template=request.correction_prompt_template,
            response=response.response,
            retries=retries,
            validation_result=validation_result,
            include_reasoning=request.include_reasoning,
            reasoning=response.reasoning,
            context_length_exceeded=response.context_length_exceeded,
            created_at=datetime.now()
        )

@dataclass
class Concept:
    id: str
    name: str
    labels: list[str]

@dataclass
class Entity:
    id: str
    name: str
    labels: list[str]

@dataclass
class Relation:
    source_id: str
    target_id: str
    label: str
    name: Optional[str] = None
