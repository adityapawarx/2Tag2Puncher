"""Filesystem layout and serving configuration.

Every path the pipeline touches is derived from ``TAGPUNCHER_DATA_DIR`` so the
same code runs in Colab, locally, and in a container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("TAGPUNCHER_DATA_DIR", "data"))
KAGGLE_DATASET = "wikimedia-foundation/wikipedia-structured-contents"
N_SHARDS = 37
SEED = 20250607


@dataclass(frozen=True)
class Paths:
    root: Path = DEFAULT_DATA_DIR

    @property
    def shards(self) -> Path:
        return self.root / "shards"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def shard_files(self, n_shards: int = N_SHARDS) -> list[Path]:
        candidates = [self.shards / f"enwiki_namespace_0_{i}.jsonl" for i in range(1, n_shards + 1)]
        return [p for p in candidates if p.exists()]

    def pecos_data(self, split: str) -> Path:
        return self.processed / f"{split}-data.txt"

    @property
    def labels(self) -> Path:
        return self.processed / "output-labels.txt"

    @property
    def vocabulary(self) -> Path:
        return self.processed / "links.csv"


@dataclass(frozen=True)
class ServingConfig:
    model_dir: Path
    top_k: int = 10
    min_score: float = 0.0
    max_chars: int = 200_000

    @classmethod
    def from_env(cls) -> ServingConfig:
        return cls(
            model_dir=Path(
                os.environ.get("TAGPUNCHER_MODEL_DIR", str(Paths().artifacts / "current"))
            ),
            top_k=int(os.environ.get("TAGPUNCHER_TOP_K", "10")),
            min_score=float(os.environ.get("TAGPUNCHER_MIN_SCORE", "0.0")),
        )
