from typing import List, NamedTuple, Optional, Iterator

from app.knowledge_graph import KnowledgeGraph
from app.llm_client import LlmClient
from app.request_repository import RequestRepository
from app.models import ModelRequest, ModelResponse, CachedModelRequest

class QuestionToQueryPipeline:
    class PipelineRunRequest(NamedTuple):
        model_name: str
        dataset: Optional[str] = None
        data: List[tuple[str, str]]
        system_prompt: str
        user_prompt_template: str
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
                    include_reasoning=self.include_reasoning,
                    allow_correction=self.allow_correction,
                    correction_prompt=self.correction_prompt_template.format(question=question)
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

        if not self._test_llm_client():
            raise ValueError("LLM client call failed")

    def _test_llm_client(self) -> bool:
        response = self.llm_client.call_model(
            ModelRequest(
                system_prompt="You are a helpful assistant.",
                user_prompt="What is the capital of France?",
                include_reasoning=True
            )
        )

        return response is not None

    def _retries_loop(
        self,
        request: ModelRequest,
        max_retries: int,
        use_cache: bool
    ) -> ModelResponse:
        last_request = None
        if use_cache:
            last_request = self._get_cached_request(request)

        while last_request.retries < max_retries:
            pass

    def _get_cached_request(self, request: ModelRequest) -> Optional[CachedModelRequest]:
        return self.request_repository.get_request_by_metadata(request)
