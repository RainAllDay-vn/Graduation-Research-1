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
        model_to_call = request.model_name
        if self.provider and model_to_call and not model_to_call.startswith(f"{self.provider}/"):
            model_to_call = f"{self.provider}/{model_to_call}"

        # Combine all possible parameters for formatting
        format_params = {**request.template_parameters}
        if request.question:
            format_params["question"] = request.question
        if request.previous_validation_result:
            format_params["validation_result"] = request.previous_validation_result

        messages = [
            {
                "role": "system", 
                "content": request.system_prompt_template.content.format(
                    **format_params
                )
            },
            {
                "role": "user", 
                "content": request.user_prompt_template.content.format(
                    **format_params,
                    question=request.question
                )
            }
        ]
        if request.previous_answer_prompt:
            messages.append({"role": "assistant", "content": request.previous_answer_prompt})
        if request.correction_prompt_template:
            messages.append({
                "role": "user", 
                "content": request.correction_prompt_template.content.format(
                    **format_params,
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
                model_name=request.model_name,
                response=content,
                reasoning=reasoning,
                context_length_exceeded=context_length_exceeded
            )

        except litellm.exceptions.APIError as e:
            logging.error("LiteLLM api call exception for model %s: %s", model_to_call, e)
        return None
