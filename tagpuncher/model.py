"""PECOS XR-Linear tagging model and its on-disk artifact format.

An artifact directory is self-describing:

    <artifact>/
        manifest.json      version, label/vocab sizes, training provenance, metrics
        preprocessor/      PECOS TF-IDF preprocessor
        xlinear_model/     PECOS XR-Linear model
        output_items.json  tag name per label id

Inference is a sparse mat-vec walk down the label tree: CPU-only and fast enough
to serve synchronously.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import scipy.sparse as sp
from pecos.utils import smat_util
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.xmc.xlinear.model import XLinearModel

DEFAULT_VECTORIZER_CONFIG: dict[str, Any] = {
    "type": "tfidf",
    "kwargs": {
        "min_df": 40,
        "max_features": 100000,
        "dtype": "float32",
        "stop_words": "english",
        "base_vect_configs": [
            {
                "ngram_range": [1, 2],
                "max_df_ratio": 0.90,
                "analyzer": "word",
                "sublinear_tf": True,
                "smooth_idf": True,
                "norm": "l2",
            }
        ],
    },
}

MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    """Provenance travelling with an artifact so a served model is identifiable."""

    version: str
    n_labels: int
    vocab_size: int
    trained_on: str = "unknown"
    max_chunk_words: int = 450
    metrics: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        known = {f: payload[f] for f in cls.__dataclass_fields__ if f in payload}
        return cls(**known)


@dataclass
class ScoredTag:
    tag: str
    score: float


class TagPuncherModel:
    """TF-IDF -> PIFA label embeddings -> label tree -> XR-Linear classifiers."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        xlinear_model: XLinearModel,
        tags: list[str],
        manifest: Manifest,
    ) -> None:
        self.preprocessor = preprocessor
        self.xlinear_model = xlinear_model
        self.tags = tags
        self.manifest = manifest

    @classmethod
    def train(
        cls,
        input_text_path: str | Path,
        output_text_path: str | Path,
        *,
        version: str,
        trained_on: str = "unknown",
        nr_splits: int = 8,
        threshold: float = 0.1,
        vectorizer_config: dict[str, Any] | None = None,
    ) -> TagPuncherModel:
        """Train from PECOS-format files (``label_ids<TAB>text`` and a tag list)."""
        parsed = Preprocessor.load_data_from_file(str(input_text_path), str(output_text_path))
        corpus = parsed["corpus"]
        labels = parsed["label_matrix"]
        tags = [
            line.strip() for line in Path(output_text_path).read_text(encoding="utf-8").splitlines()
        ]
        return cls.fit(
            corpus,
            labels,
            tags,
            version=version,
            trained_on=trained_on,
            nr_splits=nr_splits,
            threshold=threshold,
            vectorizer_config=vectorizer_config,
        )

    @classmethod
    def fit(
        cls,
        corpus: list[str],
        labels: sp.csr_matrix,
        tags: list[str],
        *,
        version: str,
        trained_on: str = "unknown",
        nr_splits: int = 8,
        threshold: float = 0.1,
        vectorizer_config: dict[str, Any] | None = None,
    ) -> TagPuncherModel:
        """Train from an in-memory corpus and label matrix."""
        config = vectorizer_config or DEFAULT_VECTORIZER_CONFIG
        preprocessor = Preprocessor.train(corpus, config)
        features = preprocessor.predict(corpus)

        label_features = LabelEmbeddingFactory.create(labels, features, method="pifa")
        cluster_chain = Indexer.gen(label_features, nr_splits=nr_splits)
        xlinear_model = XLinearModel.train(
            features,
            labels,
            C=cluster_chain,
            negative_sampling_scheme="tfn",
            threshold=threshold,
            threads=-1,
        )

        manifest = Manifest(
            version=version,
            n_labels=labels.shape[1],
            vocab_size=features.shape[1],
            trained_on=trained_on,
        )
        return cls(preprocessor, xlinear_model, tags, manifest)

    def predict_matrix(self, corpus: list[str]) -> sp.csr_matrix:
        """Score every chunk against every tag, sorted by score within each row."""
        features = self.preprocessor.predict(corpus)
        return smat_util.sorted_csr(self.xlinear_model.predict(features))

    def predict(
        self, corpus: list[str], *, top_k: int = 10, min_score: float = 0.0
    ) -> list[list[ScoredTag]]:
        """Return the ranked tags for each chunk of pre-cleaned text."""
        scores = self.predict_matrix(corpus)
        results: list[list[ScoredTag]] = []
        for row in range(scores.shape[0]):
            start, end = scores.indptr[row], scores.indptr[row + 1]
            row_tags = [
                ScoredTag(self.tags[scores.indices[i]], float(scores.data[i]))
                for i in range(start, end)
                if scores.data[i] >= min_score
            ]
            results.append(row_tags[:top_k])
        return results

    def save(self, artifact_dir: str | Path) -> Path:
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.preprocessor.save(str(path / "preprocessor"))
        self.xlinear_model.save(str(path / "xlinear_model"))
        (path / "output_items.json").write_text(json.dumps(self.tags), encoding="utf-8")
        (path / MANIFEST_NAME).write_text(self.manifest.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, artifact_dir: str | Path) -> TagPuncherModel:
        path = Path(artifact_dir)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{path} is not a TagPuncher artifact: no {MANIFEST_NAME}. "
                "Train one with `tagpuncher train` or point TAGPUNCHER_MODEL_DIR elsewhere."
            )
        preprocessor = Preprocessor.load(str(path / "preprocessor"))
        xlinear_model = XLinearModel.load(str(path / "xlinear_model"))
        tags = json.loads((path / "output_items.json").read_text(encoding="utf-8"))
        manifest = Manifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        return cls(preprocessor, xlinear_model, tags, manifest)
