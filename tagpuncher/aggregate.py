"""Roll per-chunk predictions up to a document-level tag list.

The model only ever saw ~450-word passages, so a long document is tagged chunk
by chunk and the chunk scores are combined here rather than by feeding the whole
document through TF-IDF at once (which would be out of distribution).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import ScoredTag


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
    passage a tag came from.
    """
    best: dict[str, float] = {}
    support: dict[str, list[int]] = defaultdict(list)

    for index, tags in enumerate(chunk_tags):
        for scored in tags:
            support[scored.tag].append(index)
            if scored.score > best.get(scored.tag, float("-inf")):
                best[scored.tag] = scored.score

    ranked = [
        DocumentTag(tag=tag, score=score, chunks=sorted(support[tag]))
        for tag, score in best.items()
        if score >= min_score
    ]
    ranked.sort(key=lambda t: (-t.score, t.tag))
    return ranked[:top_k]
