from typing import List, NamedTuple, Optional, Iterator

from app.knowledge_graph import KnowledgeGraph
from app.llm_client import LlmClient
from app.request_repository import RequestRepository
from app.models import ModelRequest, ModelResponse, CachedModelRequest
from app.validator import validate_query

class QuestionToQueryPipeline:
    class PipelineRunRequest(NamedTuple):
        data: List[tuple[str, str]]
        system_prompt: str
        user_prompt_template: str
        model_name: Optional[str] = None
        dataset: Optional[str] = None
        type: str = "QUESTION_TO_QUERY"
        include_reasoning: bool = True
        allow_correction: bool = True
        correction_prompt_template: str = ""
        max_retries: int = 5
        use_cache: bool = True

        def to_model_request(self) -> Iterator[ModelRequest]:
            for question, _ in self.data:
                yield ModelRequest(
                    model_name=self.model_name,
                    dataset=self.dataset,
                    question=question,
                    system_prompt=self.system_prompt,
                    user_prompt=self.user_prompt_template.format(question=question),
                    type = self.type,
                    include_reasoning=self.include_reasoning
                )

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        llm_client: LlmClient,
        request_repository: Optional[RequestRepository] = None
    ):
        self.knowledge_graph = knowledge_graph
        self.llm_client = llm_client
        self.request_repository = request_repository

    def run(
        self,
        request: PipelineRunRequest
    ):
        if request.use_cache and self.request_repository is None:
            raise ValueError("Request repository is not initialized")

        if not self._test_llm_client(request):
            raise ValueError("LLM client call failed")


    def _test_llm_client(self, request: PipelineRunRequest) -> bool:
        response = self.llm_client.call_model(
            ModelRequest(
                model_name=request.model_name,
                system_prompt="You are a helpful assistant.",
                user_prompt="What is the capital of France?",
                include_reasoning=request.include_reasoning
            )
        )

        return response is not None

    def _retries_loop(
        self,
        pipeline_request: PipelineRunRequest,
        model_request: ModelRequest
    ) -> ModelResponse:
        last_request = None
        if pipeline_request.use_cache:
            last_request = self._get_cached_request(model_request)
        current_retries = last_request.retries+1 if last_request else 0

        while current_retries < pipeline_request.max_retries:
            if last_request is not None:
                model_request.previous_answer_prompt = last_request.response
                model_request.correction_prompt = (
                    pipeline_request
                    .correction_prompt_template
                    .format(valiation_result=last_request.valiation_result)
                )
            response = self.llm_client.call_model(model_request)
            valiation_result = validate_query(self.knowledge_graph, response.response)

            last_request = CachedModelRequest.from_request_and_response(
                model_request,
                response,
                current_retries,
                valiation_result = None if valiation_result == "OK" else valiation_result
            )
            self.request_repository.save_request(last_request)
            if valiation_result == "OK":
                break
            current_retries +=1


    def _get_cached_request(self, request: ModelRequest) -> Optional[CachedModelRequest]:
        return self.request_repository.get_request_by_metadata(request)
