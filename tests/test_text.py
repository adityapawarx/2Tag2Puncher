from __future__ import annotations

from tagpuncher.text import (
    MAX_CHUNK_WORDS,
    chunk_text,
    clean_text,
    merge_passages,
    strip_pdf_artifacts,
)


def test_clean_text_matches_training_normalisation():
    assert clean_text("Naïve Bayes,  APPLIED!\n\nTo Text") == "naive bayes applied to text"


def test_merge_passages_groups_up_to_the_limit():
    passages = ["word " * 300, "word " * 100, "word " * 200]
    assert list(merge_passages(passages, max_words=450)) == [[0, 1], [2]]


def test_chunk_text_never_exceeds_the_training_chunk_size():
    document = "\n\n".join(["sentence " * 200] * 5)
    chunks = chunk_text(document)
    assert chunks
    assert all(len(chunk.split()) <= MAX_CHUNK_WORDS for chunk in chunks)


def test_chunk_text_splits_a_single_oversized_paragraph():
    chunks = chunk_text("word " * 1000)
    assert len(chunks) == 3
    assert all(len(chunk.split()) <= MAX_CHUNK_WORDS for chunk in chunks)


def test_strip_pdf_artifacts_removes_noise_but_keeps_paragraphs():
    raw = "Trans-\nformer models [12] cost $O(n^2)$.\n\nSee arXiv:2401.01234 for details."
    stripped = strip_pdf_artifacts(raw)
    assert "Transformer" in stripped
    assert "[12]" not in stripped
    assert "arXiv" not in stripped
    assert "\n\n" in stripped
