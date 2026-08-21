"""Traversal helpers for the Wikipedia structured-contents JSONL schema.

Each line of a shard is one article with a nested ``sections`` tree whose nodes
carry ``has_parts``, ``links`` and ``images``. These are the only functions that
know that schema.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import orjson

TEXT_NODE_TYPES = {"paragraph", "list_item"}


def parse_json(value: Any) -> Any:
    """Some shards store ``sections``/``infoboxes`` as embedded JSON strings."""
    if isinstance(value, str):
        try:
            return orjson.loads(value)
        except orjson.JSONDecodeError:
            return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def extract_links(node: Any) -> list[str]:
    """Collect every link URL and image URL reachable from ``node``."""
    links: set[str] = set()
    stack = [node]

    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for link in current.get("links", []) or []:
                url = link.get("url")
                if url:
                    links.add(url)
                for image in link.get("images", []) or []:
                    image_url = image.get("content_url")
                    if image_url:
                        links.add(image_url)
            for image in current.get("images", []) or []:
                image_url = image.get("content_url")
                if image_url:
                    links.add(image_url)
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)

    return sorted(links)


def count_words(node: Any) -> int:
    """Total words across every ``value`` field in the section tree."""
    total = 0
    queue = deque([node])

    while queue:
        current = queue.popleft()
        if isinstance(current, dict):
            value = current.get("value")
            if isinstance(value, str):
                total += len(value.split())
            queue.extend(current.get("has_parts", []) or [])
        elif isinstance(current, list):
            queue.extend(current)

    return total


def flatten_sections(section: Any, path: list[str] | None = None) -> list[dict[str, Any]]:
    """Flatten a section tree into its text-bearing leaves.

    Each leaf keeps a breadcrumb ``path`` (e.g. ``History > Early years``) and
    the originating ``node`` so its links can be read as that passage's labels.
    """
    if not isinstance(section, dict):
        return []

    path = list(path or [])
    name = section.get("name") or section.get("type")
    if name:
        path.append(name)

    value = section.get("value")
    if section.get("type") in TEXT_NODE_TYPES and value:
        return [{"path": " > ".join(path), "text": value.strip(), "node": section}]

    leaves: list[dict[str, Any]] = []
    for child in section.get("has_parts", []) or []:
        leaves.extend(flatten_sections(child, path))
    return leaves
