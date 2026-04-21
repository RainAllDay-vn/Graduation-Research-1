import logging
from typing import Dict, Optional
from app.models import ModelRequest, ModelResponse
from app.llm_client import LlmClient

logger = logging.getLogger(__name__)

class MockLlmClient(LlmClient):
    """
    A mock client that simulates LlmClient logic without making actual API calls.
    It can be configured with predefined responses for specific questions.
    """
    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "MATCH (n) RETURN n LIMIT 10",
        model_name: str = "mock-model"
    ):
        super().__init__(model_name=model_name)
        self.responses = responses or {}
        self.default_response = default_response
        self.last_messages = []

    def call_model(self, request: ModelRequest) -> ModelResponse:
        # Simulate message construction logic from LlmClient
        messages = [
            {"role": "system", "content": request.system_prompt.content},
            {"role": "user", "content": request.user_prompt_template.content.format(question=request.question)}
        ]
        
        if request.previous_answer_prompt:
            messages.append({"role": "assistant", "content": request.previous_answer_prompt})
            
        if request.correction_prompt_template:
            messages.append({
                "role": "user", 
                "content": request.correction_prompt_template.content.format(
                    validation_result=request.previous_validation_result
                )
            })

        self.last_messages = messages
        
        # Log the messages as if we were calling the model
        logger.info("MockLlmClient simulating call with messages: %s", messages)

        # Determine response
        response_text = self.responses.get(request.question, self.default_response)
        
        # Handle reasoning if requested
        reasoning = None
        if request.include_reasoning:
            reasoning = f"Simulated reasoning for: {request.question}"

        return ModelResponse(
            model_name=self.model_name,
            response=response_text,
            reasoning=reasoning,
            context_length_exceeded=False
        )
