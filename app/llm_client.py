import os
import logging
import litellm
from dotenv import load_dotenv

from app.models import ModelRequest, ModelResponse

load_dotenv()

# Configure LiteLLM
litellm.request_timeout = None
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

class LlmClient:
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        api_base: str = None,
        provider: str = None
    ):
        self.model_name = model_name or os.environ.get("LITELLM_MODEL")
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY")
        self.api_base = api_base or os.environ.get("LITELLM_API_URL")
        self.provider = provider or os.environ.get("LITELLM_PROVIDER")

    def call_model(self, request: ModelRequest) -> ModelResponse:
        if request.model_name is None:
            request.model_name = self.model_name

        model_to_call = request.model_name
        if self.provider and model_to_call and not model_to_call.startswith(f"{self.provider}/"):
            model_to_call = f"{self.provider}/{model_to_call}"

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

        kwargs = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if request.include_reasoning:
            # LiteLLM's standard reasoning trigger
            kwargs["include_reasoning"] = True
            # SGLang specific trigger for Qwen/DeepSeek
            kwargs["extra_body"] = {"enable_thinking": True}

        try:
            response = litellm.completion(
                model=model_to_call,
                messages=messages,
                api_key=self.api_key,
                **kwargs
            )

            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None)
            finish_reason = getattr(response.choices[0], "finish_reason", "")
            context_length_exceeded = finish_reason == "length"

            return ModelResponse(
                model_name=model_to_call,
                response=content,
                reasoning=reasoning,
                context_length_exceeded=context_length_exceeded
            )

        except litellm.exceptions.APIError as e:
            logging.error("LiteLLM api call exception for model %s: %s", model_to_call, e)
        return None
