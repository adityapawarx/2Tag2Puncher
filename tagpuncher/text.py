"""Text normalisation and chunking.

Both training and serving import from this module. Any divergence between how a
document is prepared for training and how it is prepared for inference shows up
as silent accuracy loss, so there is exactly one implementation of each step.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator

MAX_CHUNK_WORDS = 450

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?")
_INLINE_MATH = re.compile(r"\$[^$]*\$")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_CITATION = re.compile(r"\[\s*\d+(\s*[,-]\s*\d+)*\s*\]")
_ARXIV_ID = re.compile(r"arxiv\s*:\s*\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Normalise a passage the way the training corpus was normalised."""
    lowered = text.lower()
    without_punctuation = _PUNCTUATION.sub(" ", lowered)
    collapsed = _WHITESPACE.sub(" ", without_punctuation).strip()
    return strip_accents(collapsed)


def strip_pdf_artifacts(text: str) -> str:
    """Remove PDF/LaTeX noise while preserving paragraph structure.

    The model was trained on Wikipedia prose, so citation markers, math and
    hyphenated line breaks are all out of distribution; dropping them moves an
    extracted paper closer to the training domain. Blank lines survive so the
    result can still be chunked by paragraph.
    """
    dehyphenated = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    without_math = _INLINE_MATH.sub(" ", dehyphenated)
    without_commands = _LATEX_COMMAND.sub(" ", without_math)
    without_citations = _CITATION.sub(" ", without_commands)
    return _ARXIV_ID.sub(" ", without_citations)


def word_count(text: str) -> int:
    return len(text.split())


def merge_passages(
    passages: Iterable[str], max_words: int = MAX_CHUNK_WORDS
) -> Iterator[list[int]]:
    """Group consecutive passages into chunks of at most ``max_words`` words.

    Yields lists of indices into ``passages`` so callers can carry per-passage
    metadata (labels during training, offsets during serving) through the merge.
    A passage longer than ``max_words`` becomes a chunk of its own.
    """
    group: list[int] = []
    group_words = 0

    for index, passage in enumerate(passages):
        words = word_count(passage)
        if group and group_words + words > max_words:
            yield group
            group, group_words = [], 0
        group.append(index)
        group_words += words

    if group:
        yield group


def split_long_passage(passage: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split an oversized passage into word windows of at most ``max_words``."""
    words = passage.split()
    if len(words) <= max_words:
        return [passage]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split a document into cleaned chunks sized like the training examples."""
    paragraphs = [
        window
        for block in re.split(r"\n\s*\n", text)
        if block.strip()
        for window in split_long_passage(block.strip(), max_words)
    ]
    if not paragraphs:
        return []

    chunks = []
    for indices in merge_passages(paragraphs, max_words):
        cleaned = clean_text(" ".join(paragraphs[i] for i in indices))
        if cleaned:
            chunks.append(cleaned)
    return chunks
