import re

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from dialog_model.language_generator.generator import ResponseCandidatesGenerator


class ResponseCandidates(BaseModel):
    response_candidates: List[str] = Field()


class ResponseCandidatesParams(BaseModel):
    context: List[str] = Field()
    n_candidates: int = Field(default=20, ge=1, le=64)
    max_n_context_tokens: int = Field(default=70, ge=1, le=90)
    temperature: float = Field(default=0.7, gt=0, le=100)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=5.0, ge=1.0, le=100)


class EndpointsRegister:
    def __init__(self, app: FastAPI, generator: ResponseCandidatesGenerator):
        self._app = app
        self._generator = generator

    def register_response_candidates_view(self):
        @self._app.post("/response_candidates/", response_model=ResponseCandidates)
        def get_response_candidates(params: ResponseCandidatesParams) -> ResponseCandidates:
            candidates = self._generator(
                context=params.context,
                n_candidates=params.n_candidates,
                max_n_context_tokens=params.max_n_context_tokens,
                repetition_penalty=params.repetition_penalty,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p)

            return ResponseCandidates(response_candidates=candidates)

        return get_response_candidates

    def register_health_check_view(self):
        @self._app.get("/health_check/")
        def health_check():
            return "ok"

        return health_check

    def register_all_views(self):
        for field in dir(self):
            if re.match('^register.+view$', field):
                getattr(self, field)()
