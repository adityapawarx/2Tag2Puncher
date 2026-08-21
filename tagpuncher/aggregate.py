"""Roll per-chunk predictions up to a document-level tag list.

The model only ever saw ~450-word passages, so a long document is tagged chunk
by chunk and the chunk scores are combined here rather than by feeding the whole
document through TF-IDF at once (which would be out of distribution).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import ScoredTag

#: A chunk only counts as supporting a tag if it scores at least this fraction of
#: the tag's best chunk. PECOS returns a score for every requested tag, most of
#: them ~0, so without this every tag would claim every chunk.
SUPPORT_RATIO = 0.1


@dataclass
class DocumentTag:
    tag: str
    score: float
    chunks: list[int]


def aggregate(
    chunk_tags: list[list[ScoredTag]],
    *,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[DocumentTag]:
    """Combine chunk-level tags by taking each tag's best chunk score.

    Max rather than sum: summing rewards tags that appear weakly in many chunks,
    which in practice surfaces generic filler tags over the specific ones.
    ``chunks`` records the supporting chunk indices so a UI can highlight the
    passage a tag came from; a chunk supports a tag only if it scores within
    ``SUPPORT_RATIO`` of that tag's best chunk.
    """
    best: dict[str, float] = {}
    scores: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for index, tags in enumerate(chunk_tags):
        for scored in tags:
            scores[scored.tag].append((index, scored.score))
            if scored.score > best.get(scored.tag, float("-inf")):
                best[scored.tag] = scored.score

    ranked = [
        DocumentTag(tag=tag, score=score, chunks=_support(scores[tag], score))
        for tag, score in best.items()
        if score >= min_score
    ]
    ranked.sort(key=lambda t: (-t.score, t.tag))
    return ranked[:top_k]


def _support(chunk_scores: list[tuple[int, float]], best: float) -> list[int]:
    cutoff = max(best * SUPPORT_RATIO, 1e-9)
    return sorted(index for index, score in chunk_scores if score >= cutoff)
