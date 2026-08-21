from __future__ import annotations

from tagpuncher.aggregate import aggregate
from tagpuncher.model import ScoredTag


def test_aggregate_takes_the_best_chunk_score_and_records_support():
    chunk_tags = [
        [ScoredTag("Astronomy", 0.9), ScoredTag("Physics", 0.4)],
        [ScoredTag("Astronomy", 0.3), ScoredTag("Physics", 0.8)],
    ]
    tags = aggregate(chunk_tags, top_k=5)

    assert [t.tag for t in tags] == ["Astronomy", "Physics"]
    assert tags[0].score == 0.9
    assert tags[0].chunks == [0, 1]


def test_aggregate_excludes_chunks_that_barely_score_the_tag():
    chunk_tags = [
        [ScoredTag("Astronomy", 1.0), ScoredTag("Music", 0.0)],
        [ScoredTag("Astronomy", 0.0), ScoredTag("Music", 1.0)],
    ]
    tags = {t.tag: t.chunks for t in aggregate(chunk_tags, top_k=5)}

    assert tags == {"Astronomy": [0], "Music": [1]}


def test_aggregate_applies_min_score_and_top_k():
    chunk_tags = [[ScoredTag("A", 0.9), ScoredTag("B", 0.5), ScoredTag("C", 0.1)]]
    assert [t.tag for t in aggregate(chunk_tags, top_k=2)] == ["A", "B"]
    assert [t.tag for t in aggregate(chunk_tags, min_score=0.6)] == ["A"]


def test_aggregate_handles_no_predictions():
    assert aggregate([[], []]) == []
