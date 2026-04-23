import os
import threading
import logging
import hashlib
from datetime import datetime
from typing import List, NamedTuple, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, Future
import dotenv

from app.knowledge_graph import KnowledgeGraph
from app.llm_client import LlmClient
from app.request_repository import RequestRepository
from app.models import (
    ModelRequest,
    ModelResponse,
    SystemPromptTemplate,
    UserPromptTemplate,
    CorrectionPromptTemplate
)
from app.validator import validate_query

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

class QuestionToQueryPipeline:
    class PipelineRunRequest(NamedTuple):
        data: List[tuple[str, str]]
        system_prompt_template: str
        user_prompt_template: str
        model_name: str = os.getenv("LITELLM_MODEL")
        template_parameters: dict[str, str] = {}
        dataset: Optional[str] = None
        type: str = "QUESTION_TO_QUERY"
        include_reasoning: bool = True
        allow_correction: bool = True
        correction_prompt_template: str = ""
        max_retries: int = 5
        use_cache: bool = True

        def to_model_request(self) -> Iterator[ModelRequest]:
            system_prompt_template_obj = SystemPromptTemplate(
                id=hashlib.sha256(self.system_prompt_template.encode()).hexdigest()[:16],
                content=self.system_prompt_template,
                created_at=datetime.now()
            )
            user_prompt_template_obj = UserPromptTemplate(
                id=hashlib.sha256(self.user_prompt_template.encode()).hexdigest()[:16],
                content=self.user_prompt_template,
                created_at=datetime.now()
            )
            for question, _ in self.data:
                yield ModelRequest(
                    model_name=self.model_name,
                    dataset=self.dataset,
                    question=question,
                    system_prompt_template=system_prompt_template_obj,
                    user_prompt_template=user_prompt_template_obj,
                    template_parameters=self.template_parameters,
                    type = self.type,
                    include_reasoning=self.include_reasoning
                )

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        llm_client: LlmClient,
        request_repository: Optional[RequestRepository] = None,
        workers: int = os.getenv("LITELLM_WORKERS") or 10
    ):
        self.knowledge_graph = knowledge_graph
        self.llm_client = llm_client
        self.request_repository = request_repository
        self.workers = workers
        self.progress_lock = threading.Lock()
        self.progress = 0

        print(self.workers)

    def run(
        self,
        request: PipelineRunRequest
    ):
        logger.info("Starting question to query pipeline with %d questions...", len(request.data))
        if request.use_cache and self.request_repository is None:
            raise ValueError("Request repository is not initialized")

        logger.info("Testing LLM client call...")
        if not self._test_llm_client(request):
            raise ValueError("LLM client test call failed, the model may not be available")

        logger.info("Starting the generation loop...")
        def handle_result(future: Future):
            try:
                future.result()
            except Exception as e:
                logger.error("Thread generated an exception: %s", e, exc_info=True)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for model_request in request.to_model_request():
                if request.allow_correction:
                    future = executor.submit(self._retries_loop, request, model_request)
                else:
                    future = executor.submit(self._one_shot_generation, request, model_request)
                future.add_done_callback(handle_result)

    def _increase_progress(self):
        with self.progress_lock:
            self.progress += 1
            logger.info("Progress: %d requests processed...", self.progress)

    def _test_llm_client(self, request: PipelineRunRequest) -> bool:
        try:
            _ = self.llm_client.call_model(
            ModelRequest(
                model_name=request.model_name,
                system_prompt_template=SystemPromptTemplate(
                    id="test_system",
                    content="You are a helpful assistant.",
                    created_at=datetime.now()
                ),
                user_prompt_template=UserPromptTemplate(
                    id="test_user",
                    content="What is the capital of France?",
                    created_at=datetime.now()
                ),
                template_parameters={},
                question="What is the capital of France?",
                include_reasoning=request.include_reasoning
            )
        )
            return True
        except Exception as _:
            return False

    def _retries_loop(
        self,
        pipeline_request: PipelineRunRequest,
        model_request: ModelRequest
    ) -> ModelResponse:
        last_request = None
        if pipeline_request.use_cache:
            last_request = self.request_repository.get_request_by_metadata(model_request)
            if last_request is not None:
                logger.info(
                    "Using cached request for question: %s (Retries: %d)",
                    model_request.question, last_request.retries
                )
        current_retries = last_request.retries+1 if last_request else 0

        while current_retries < pipeline_request.max_retries:
            logger.info(
                "Processing request for question: %s (Retries: %d)",
                model_request.question, current_retries
            )
            if last_request is not None:
                model_request.previous_answer_prompt = last_request.response
                model_request.previous_request_id = last_request.id
                model_request.previous_validation_result = last_request.validation_result
                model_request.correction_prompt_template = CorrectionPromptTemplate(
                    id=(hashlib
                        .sha256(pipeline_request.correction_prompt_template.encode())
                        .hexdigest()[:16]),
                    content=pipeline_request.correction_prompt_template,
                    created_at=datetime.now()
                )
            response = self.llm_client.call_model(model_request)
            validation_result = validate_query(self.knowledge_graph, response.response)

            last_request = self.request_repository.save_request_from_model_request_and_response(
                model_request,
                response,
                current_retries,
                validation_result
            )
            if validation_result == "OK":
                break
            current_retries +=1

        self._increase_progress()
        return response

    def _one_shot_generation(
        self,
        pipeline_request: PipelineRunRequest,
        model_request: ModelRequest
    ) -> ModelResponse:
        if pipeline_request.use_cache:
            last_request = self.request_repository.get_request_by_metadata(model_request)
            if last_request is not None:
                logger.info("Using cached request for question: %s", model_request.question)
                return last_request

        response = self.llm_client.call_model(model_request)
        validation_result = validate_query(self.knowledge_graph, response.response)
        self.request_repository.save_request_from_model_request_and_response(
            model_request,
            response,
            0,
            validation_result
        )
        self._increase_progress()
        return response
