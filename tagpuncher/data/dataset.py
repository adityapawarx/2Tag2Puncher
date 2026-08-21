"""Turn selected articles into ~450-word training chunks in PECOS format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..text import MAX_CHUNK_WORDS, clean_text, merge_passages, word_count
from .vocabulary import iter_articles, title_from_url
from .wiki import extract_links, flatten_sections, parse_json


@dataclass
class Chunk:
    identifier: str
    title: str
    section_path: str
    text: str
    tags: list[str]

    @property
    def length(self) -> int:
        return word_count(self.text)


def article_chunks(
    article: dict[str, Any],
    url_to_title: dict[str, str],
    max_words: int = MAX_CHUNK_WORDS,
) -> list[Chunk]:
    """Flatten one article into merged, label-carrying chunks.

    A leaf passage's labels are the wiki-links inside it plus the article's
    infobox links; merged passages take the union.
    """
    sections = parse_json(article.get("sections", [])) or []
    if not isinstance(sections, list):
        sections = [sections]

    infobox_tags = _titles(
        extract_links(parse_json(article.get("infoboxes", [])) or []), url_to_title
    )

    leaves: list[dict[str, Any]] = []
    for section in sections:
        leaves.extend(flatten_sections(section))
    leaves = [leaf for leaf in leaves if leaf["text"]]
    if not leaves:
        return []

    texts = [leaf["text"] for leaf in leaves]
    chunks = []
    for indices in merge_passages(texts, max_words):
        tags = set(infobox_tags)
        for i in indices:
            tags.update(_titles(extract_links(leaves[i]["node"]), url_to_title))
        chunks.append(
            Chunk(
                identifier=str(article.get("identifier")),
                title=str(article.get("name", "")),
                section_path=leaves[indices[0]]["path"],
                text=" ".join(texts[i] for i in indices),
                tags=sorted(tags),
            )
        )
    return chunks


def _titles(urls: list[str], url_to_title: dict[str, str]) -> list[str]:
    titles = []
    for url in urls:
        title = url_to_title.get(url) or title_from_url(url)
        if title:
            titles.append(title)
    return titles


def build_chunks(
    shard_paths: list[Path],
    identifiers: set[str],
    url_to_title: dict[str, str],
    max_words: int = MAX_CHUNK_WORDS,
) -> list[Chunk]:
    """Collect chunks for the articles in one split."""
    remaining = set(identifiers)
    chunks: list[Chunk] = []

    for shard in shard_paths:
        if not remaining:
            break
        for article in iter_articles(shard):
            identifier = str(article.get("identifier"))
            if identifier not in remaining:
                continue
            remaining.discard(identifier)
            chunks.extend(article_chunks(article, url_to_title, max_words))

    return chunks


def encode_tags(chunks: list[Chunk], vocabulary: set[str]) -> tuple[list[str], list[list[int]]]:
    """Restrict tags to the vocabulary and assign contiguous label ids.

    Ids are assigned by descending training frequency so the label tree is built
    over a stable, meaningful ordering.
    """
    frequency: dict[str, int] = {}
    for chunk in chunks:
        for tag in chunk.tags:
            if tag in vocabulary:
                frequency[tag] = frequency.get(tag, 0) + 1

    tags = sorted(frequency, key=lambda t: (-frequency[t], t))
    tag_ids = {tag: index for index, tag in enumerate(tags)}
    encoded = [sorted(tag_ids[t] for t in chunk.tags if t in tag_ids) for chunk in chunks]
    return tags, encoded


def write_pecos_files(
    chunks: list[Chunk],
    encoded: list[list[int]],
    data_path: Path,
    *,
    clean: bool = True,
) -> int:
    """Write ``label_ids<TAB>text`` lines, skipping chunks with no labels."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with data_path.open("w", encoding="utf-8") as handle:
        for chunk, label_ids in zip(chunks, encoded, strict=True):
            if not label_ids:
                continue
            text = clean_text(chunk.text) if clean else chunk.text.replace("\n", " ").strip()
            if not text:
                continue
            handle.write(f"{','.join(map(str, label_ids))}\t{text}\n")
            written += 1

    return written


def write_label_file(tags: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tags) + "\n", encoding="utf-8")
