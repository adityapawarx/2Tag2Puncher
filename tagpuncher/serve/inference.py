"""Document-level tagging on top of the chunk-level model."""

from __future__ import annotations

from dataclasses import dataclass

from ..aggregate import DocumentTag, aggregate
from ..model import TagPuncherModel
from ..text import chunk_text, strip_pdf_artifacts


@dataclass
class TaggedDocument:
    tags: list[DocumentTag]
    chunks: list[str]
    model_version: str


class TaggingService:
    """Holds the loaded model; created once per process, never per request."""

    def __init__(self, model: TagPuncherModel) -> None:
        self.model = model

    @property
    def model_version(self) -> str:
        return self.model.manifest.version

    def tag(
        self,
        text: str,
        *,
        top_k: int = 10,
        min_score: float = 0.0,
        from_pdf: bool = False,
    ) -> TaggedDocument:
        chunks = chunk_text(strip_pdf_artifacts(text) if from_pdf else text)
        if not chunks:
            return TaggedDocument(tags=[], chunks=[], model_version=self.model_version)

        # Chunks are scored in a single batched call: one sparse mat-vec over the
        # whole batch is far cheaper than one call per chunk.
        per_chunk = self.model.predict(chunks, top_k=max(top_k * 3, top_k))
        tags = aggregate(per_chunk, top_k=top_k, min_score=min_score)
        return TaggedDocument(tags=tags, chunks=chunks, model_version=self.model_version)


def extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF; raises ImportError if the serve extras are absent."""
    import fitz  # PyMuPDF

    with fitz.open(stream=data, filetype="pdf") as document:
        return "\n\n".join(page.get_text() for page in document)
