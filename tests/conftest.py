from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tagpuncher.model import TagPuncherModel

TOPICS: dict[str, list[str]] = {
    "Astronomy": [
        "the telescope observed a distant galaxy and its orbiting stars",
        "astronomers measured the orbit of the comet around the star",
        "a supernova brightened the night sky above the observatory",
    ],
    "Chemistry": [
        "the reaction produced a precipitate in the aqueous solution",
        "chemists measured the enthalpy of the exothermic reaction",
        "an acid and a base neutralise to form a salt and water",
    ],
    "Music": [
        "the orchestra performed a symphony in three movements",
        "the guitarist tuned the strings before the concert began",
        "a choir sang the chorus over a slow piano melody",
    ],
}

TINY_VECTORIZER_CONFIG = {
    "type": "tfidf",
    "kwargs": {
        "min_df": 1,
        "max_features": 5000,
        "dtype": "float32",
        "base_vect_configs": [
            {
                "ngram_range": [1, 2],
                "max_df_ratio": 1.0,
                "analyzer": "word",
                "sublinear_tf": True,
                "smooth_idf": True,
                "norm": "l2",
            }
        ],
    },
}


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> TagPuncherModel:
    """A real (very small) PECOS model, so serving is exercised end to end."""
    tags = sorted(TOPICS)
    corpus: list[str] = []
    rows: list[int] = []
    columns: list[int] = []

    for label_id, tag in enumerate(tags):
        for sentence in TOPICS[tag]:
            rows.append(len(corpus))
            columns.append(label_id)
            corpus.append(sentence)

    labels = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(corpus), len(tags)),
    )

    return TagPuncherModel.fit(
        corpus,
        labels,
        tags,
        version="test",
        trained_on="synthetic",
        nr_splits=2,
        threshold=0.0,
        vectorizer_config=TINY_VECTORIZER_CONFIG,
    )
