from __future__ import annotations

from tagpuncher.data.dataset import article_chunks, encode_tags, write_pecos_files
from tagpuncher.data.wiki import count_words, extract_links, flatten_sections

ARTICLE = {
    "identifier": "42",
    "name": "Telescope",
    "sections": [
        {
            "name": "History",
            "type": "section",
            "has_parts": [
                {
                    "type": "paragraph",
                    "value": "An early telescope observed Jupiter.",
                    "links": [{"url": "https://en.wikipedia.org/wiki/Jupiter"}],
                },
                {
                    "type": "list_item",
                    "value": "Refracting designs used lenses.",
                    "links": [{"url": "https://en.wikipedia.org/wiki/Lens"}],
                },
            ],
        }
    ],
    "infoboxes": [{"links": [{"url": "https://en.wikipedia.org/wiki/Optics"}]}],
}


def test_flatten_sections_keeps_breadcrumbs():
    leaves = flatten_sections(ARTICLE["sections"][0])
    assert [leaf["path"] for leaf in leaves] == [
        "History > paragraph",
        "History > list_item",
    ]


def test_extract_links_and_word_count_walk_the_tree():
    assert extract_links(ARTICLE["sections"]) == [
        "https://en.wikipedia.org/wiki/Jupiter",
        "https://en.wikipedia.org/wiki/Lens",
    ]
    assert count_words(ARTICLE["sections"]) == 9


def test_article_chunks_merge_passages_and_union_tags():
    chunks = article_chunks(ARTICLE, url_to_title={})
    assert len(chunks) == 1
    assert chunks[0].tags == ["Jupiter", "Lens", "Optics"]
    assert chunks[0].section_path == "History > paragraph"


def test_encode_tags_orders_ids_by_frequency_and_drops_unknown_tags(tmp_path):
    chunks = article_chunks(ARTICLE, url_to_title={})
    chunks = chunks + chunks
    tags, encoded = encode_tags(chunks, vocabulary={"Jupiter", "Optics"})

    assert tags == ["Jupiter", "Optics"]
    assert encoded == [[0, 1], [0, 1]]

    path = tmp_path / "training-data.txt"
    assert write_pecos_files(chunks, encoded, path) == 2
    first = path.read_text(encoding="utf-8").splitlines()[0]
    ids, text = first.split("\t")
    assert ids == "0,1"
    assert text == "an early telescope observed jupiter refracting designs used lenses"
