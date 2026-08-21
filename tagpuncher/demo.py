"""A placeholder artifact for wiring up the stack without the real model.

The Wikipedia-trained artifact is hundreds of megabytes and takes hours to
reproduce. This trains a genuine (but tiny) PECOS model over a handful of
sentences per topic, so the API, the UI and a deployment can be exercised
end to end. Its predictions are illustrative only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .model import TagPuncherModel

DEMO_TOPICS: dict[str, list[str]] = {
    "Machine_learning": [
        "a neural network is trained with gradient descent on labelled examples",
        "the classifier generalises to unseen data after regularisation",
        "supervised learning minimises a loss over a training set",
    ],
    "Information_retrieval": [
        "the search engine ranks documents by their relevance to a query",
        "an inverted index maps terms to the documents that contain them",
        "tf idf weighting downweights terms that occur in many documents",
    ],
    "Natural_language_processing": [
        "the model tokenises text and predicts the next word in a sentence",
        "word embeddings place semantically similar words close together",
        "a language model assigns probabilities to sequences of tokens",
    ],
    "Astronomy": [
        "the telescope observed a distant galaxy and its orbiting stars",
        "astronomers measured the orbit of a comet around the star",
        "a supernova brightened the sky above the observatory",
    ],
    "Chemistry": [
        "the reaction produced a precipitate in the aqueous solution",
        "chemists measured the enthalpy of an exothermic reaction",
        "an acid and a base neutralise to form a salt and water",
    ],
    "Music": [
        "the orchestra performed a symphony in three movements",
        "the guitarist tuned the strings before the concert began",
        "a choir sang the chorus over a slow piano melody",
    ],
}

DEMO_VECTORIZER_CONFIG = {
    "type": "tfidf",
    "kwargs": {
        "min_df": 1,
        "max_features": 20000,
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


def build_demo_model() -> TagPuncherModel:
    tags = sorted(DEMO_TOPICS)
    corpus: list[str] = []
    rows: list[int] = []
    columns: list[int] = []

    for label_id, tag in enumerate(tags):
        for sentence in DEMO_TOPICS[tag]:
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
        version="demo",
        trained_on="bundled demo sentences (not the Wikipedia corpus)",
        nr_splits=2,
        threshold=0.0,
        vectorizer_config=DEMO_VECTORIZER_CONFIG,
    )


def write_demo_artifact(output_dir: str | Path) -> Path:
    return build_demo_model().save(output_dir)
