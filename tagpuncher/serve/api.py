"""FastAPI application exposing the tagging model."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import ServingConfig
from ..model import TagPuncherModel
from .inference import TaggedDocument, TaggingService, extract_pdf_text

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class TagRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0)


class Tag(BaseModel):
    tag: str
    score: float
    chunks: list[int]


class TagResponse(BaseModel):
    tags: list[Tag]
    chunks: list[str]
    model_version: str


class Health(BaseModel):
    status: str
    model_version: str | None
    n_labels: int | None


def to_response(document: TaggedDocument) -> TagResponse:
    return TagResponse(
        tags=[Tag(tag=t.tag, score=t.score, chunks=t.chunks) for t in document.tags],
        chunks=document.chunks,
        model_version=document.model_version,
    )


def create_app(service: TaggingService | None = None) -> FastAPI:
    """Build the app. Passing a ``service`` skips artifact loading (used in tests)."""
    config = ServingConfig.from_env()
    state: dict[str, TaggingService | None] = {"service": service}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Loading takes seconds and inference takes milliseconds, so the artifact
        # is read once per process and shared by every request.
        if state["service"] is None:
            state["service"] = TaggingService(TagPuncherModel.load(config.model_dir))
        yield

    app = FastAPI(title="TagPuncher", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_service() -> TaggingService:
        loaded = state["service"]
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return loaded

    @app.get("/v1/healthz", response_model=Health)
    def healthz() -> Health:
        loaded = state["service"]
        if loaded is None:
            return Health(status="loading", model_version=None, n_labels=None)
        return Health(
            status="ok",
            model_version=loaded.model_version,
            n_labels=loaded.model.manifest.n_labels,
        )

    @app.post("/v1/tag", response_model=TagResponse)
    def tag(request: TagRequest, service: TaggingService = Depends(get_service)) -> TagResponse:
        if len(request.text) > config.max_chars:
            raise HTTPException(status_code=413, detail="text too long")
        return to_response(
            service.tag(request.text, top_k=request.top_k, min_score=request.min_score)
        )

    @app.post("/v1/tag/document", response_model=TagResponse)
    async def tag_document(
        file: UploadFile = File(...),
        top_k: int = Form(10),
        min_score: float = Form(0.0),
        service: TaggingService = Depends(get_service),
    ) -> TagResponse:
        payload = await file.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")

        is_pdf = payload[:4] == b"%PDF"
        if is_pdf:
            try:
                text = extract_pdf_text(payload)
            except ImportError as error:  # pragma: no cover - depends on extras
                raise HTTPException(
                    status_code=501, detail="PDF support requires the 'serve' extra"
                ) from error
        else:
            text = payload.decode("utf-8", errors="replace")

        if not text.strip():
            raise HTTPException(status_code=422, detail="no extractable text")

        return to_response(service.tag(text, top_k=top_k, min_score=min_score, from_pdf=is_pdf))

    return app


app = create_app()
