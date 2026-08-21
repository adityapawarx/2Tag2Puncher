"""Build the tag vocabulary from corpus-wide wiki-link statistics."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .wiki import count_words, extract_links, parse_json

ASSET_SUFFIXES = re.compile(
    r"\.(?:svg|png|jpg|jpeg|gif|webp|asp|pdf|html|php|phtml|page|aspx)(?:\?[^/]+)?$",
    re.IGNORECASE,
)
URL_TAIL = re.compile(r"/([^/#?]+)(?:[#?].*)?$")

MIN_LINK_FREQUENCY = 50
NAVIGATIONAL_PREFIXES = ("index", "list", "alphabetical", "timeline", "iucn", "redirect")
MIN_ARTICLE_LINKS = 25
MIN_ARTICLE_WORDS = 100


def title_from_url(url: str) -> str | None:
    match = URL_TAIL.search(str(url))
    return match.group(1) if match else None


def summarise_shards(shard_paths: list[Path], output_dir: Path) -> tuple[Path, Path]:
    """One streaming pass over the shards producing per-article and per-link stats.

    Writes ``all_links_summary.csv`` (identifier, name, url, total_num_links,
    text_length, link_density) and ``unique_links_counts.csv`` (url, count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_links_summary.csv"
    counts_path = output_dir / "unique_links_counts.csv"
    counter: Counter[str] = Counter()

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["identifier", "name", "url", "total_num_links", "text_length", "link_density"]
        )
        for shard in shard_paths:
            for article in iter_articles(shard):
                sections = parse_json(article.get("sections", []))
                infoboxes = parse_json(article.get("infoboxes", []))
                links = sorted(set(extract_links(sections) + extract_links(infoboxes)))
                counter.update(links)

                words = count_words(sections)
                writer.writerow(
                    [
                        article.get("identifier"),
                        article.get("name"),
                        article.get("url", ""),
                        len(links),
                        words,
                        len(links) / words if words else 0,
                    ]
                )

    with counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["url", "count"])
        writer.writerows(counter.items())

    return summary_path, counts_path


def iter_articles(shard_path: Path):
    import orjson

    with shard_path.open("rb") as handle:
        for line in handle:
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                continue


def build_vocabulary(
    summary_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    min_frequency: int = MIN_LINK_FREQUENCY,
) -> pd.DataFrame:
    """Reduce raw link counts to the candidate tag set.

    Drops asset links, Wikipedia meta pages, and anything too rare to learn.
    """
    url_to_title = dict(zip(summary_df["url"], summary_df["name"], strict=False))
    counts = counts_df.copy()
    counts["title"] = counts["url"].map(url_to_title)

    missing = counts["title"].isna()
    counts.loc[missing, "title"] = counts.loc[missing, "url"].map(title_from_url)

    keep = (
        counts["title"].notna()
        & ~counts["title"].str.contains(ASSET_SUFFIXES, na=False, regex=True)
        & ~counts["title"].str.contains("wiki", case=False, na=False)
        & (counts["count"] >= min_frequency)
    )
    return counts.loc[keep, ["url", "title", "count"]].reset_index(drop=True)


def select_articles(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Keep densely-linked prose articles, dropping navigational pages."""
    df = summary_df.copy()
    names = df["name"].astype(str).str.lower()
    df = df[~names.str.startswith(NAVIGATIONAL_PREFIXES)]
    df = df[(df["total_num_links"] > MIN_ARTICLE_LINKS) & (df["text_length"] > MIN_ARTICLE_WORDS)]
    df = df.sort_values("total_num_links", ascending=False).drop_duplicates("url", keep="first")

    unnamed = df["name"].isna()
    df.loc[unnamed, "name"] = df.loc[unnamed, "url"].astype(str).map(title_from_url)

    return df.loc[df.groupby("identifier")["link_density"].idxmax()]


def split_articles(
    articles: pd.DataFrame, n_train: int = 100_000, n_eval: int = 1_000
) -> dict[str, pd.DataFrame]:
    """Split by link density: train on the densest pages, validate on the sparsest.

    The validation split is deliberately the low-density tail — content with
    little to no tagging, which is the case the system exists to handle.
    """
    ranked = articles.sort_values("link_density", ascending=False)
    return {
        "training": ranked.head(n_train),
        "testing": ranked.iloc[n_train : n_train + n_eval],
        "validation": ranked.tail(n_eval),
    }
