"""``tagpuncher`` command line: download -> prepare -> train -> evaluate."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import KAGGLE_DATASET, N_SHARDS, SEED, Paths
from .data.dataset import build_chunks, encode_tags, write_label_file, write_pecos_files
from .data.vocabulary import build_vocabulary, select_articles, split_articles, summarise_shards
from .model import TagPuncherModel


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def cmd_download(args: argparse.Namespace) -> None:
    import kagglehub

    paths = Paths(Path(args.data_dir))
    paths.shards.mkdir(parents=True, exist_ok=True)

    for index in range(1, args.n_shards + 1):
        name = f"enwiki_namespace_0/enwiki_namespace_0_{index}.jsonl"
        destination = paths.shards / f"enwiki_namespace_0_{index}.jsonl"
        if destination.exists():
            print(f"skip {destination.name}")
            continue
        downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET, path=name))
        downloaded.replace(destination)
        print(f"saved {destination}")


def cmd_prepare(args: argparse.Namespace) -> None:
    seed_everything()
    paths = Paths(Path(args.data_dir))
    shards = paths.shard_files(args.n_shards)
    if not shards:
        raise SystemExit(f"no shards under {paths.shards}; run `tagpuncher download` first")

    paths.processed.mkdir(parents=True, exist_ok=True)
    summary_path, counts_path = summarise_shards(shards, paths.processed)
    summary_df = pd.read_csv(summary_path)
    counts_df = pd.read_csv(counts_path)

    vocabulary_df = build_vocabulary(summary_df, counts_df)
    vocabulary_df.to_csv(paths.vocabulary, index=False)
    print(f"vocabulary: {len(vocabulary_df)} candidate tags")

    articles = select_articles(summary_df)
    splits = split_articles(articles, n_train=args.n_train, n_eval=args.n_eval)
    url_to_title = dict(zip(summary_df["url"], summary_df["name"], strict=False))
    vocabulary = set(vocabulary_df["title"])

    training_chunks = build_chunks(
        shards, set(splits["training"]["identifier"].astype(str)), url_to_title
    )
    tags, encoded = encode_tags(training_chunks, vocabulary)
    write_label_file(tags, paths.labels)
    tag_ids = {tag: index for index, tag in enumerate(tags)}
    print(f"labels: {len(tags)}")

    written = write_pecos_files(training_chunks, encoded, paths.pecos_data("training"))
    print(f"training: {written} chunks")

    for split in ("testing", "validation"):
        chunks = build_chunks(shards, set(splits[split]["identifier"].astype(str)), url_to_title)
        ids = [sorted(tag_ids[t] for t in chunk.tags if t in tag_ids) for chunk in chunks]
        written = write_pecos_files(chunks, ids, paths.pecos_data(split))
        print(f"{split}: {written} chunks")


def cmd_train(args: argparse.Namespace) -> None:
    seed_everything()
    paths = Paths(Path(args.data_dir))
    version = args.version or date.today().isoformat()

    model = TagPuncherModel.train(
        paths.pecos_data("training"),
        paths.labels,
        version=version,
        trained_on=f"{KAGGLE_DATASET} shards 1-{args.n_shards}",
        nr_splits=args.nr_splits,
    )
    destination = Path(args.output or paths.artifacts / version)
    model.save(destination)
    print(f"saved model {version} to {destination}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from pecos.utils import smat_util
    from pecos.utils.featurization.text.preprocess import Preprocessor

    paths = Paths(Path(args.data_dir))
    model = TagPuncherModel.load(args.model_dir)
    parsed = Preprocessor.load_data_from_file(str(paths.pecos_data(args.split)), str(paths.labels))
    predictions = model.predict_matrix(parsed["corpus"])
    metrics = smat_util.Metrics.generate(parsed["label_matrix"], predictions, topk=args.top_k)
    print(metrics)

    report = {f"p@{k}": round(float(p), 4) for k, p in enumerate(metrics.prec, start=1)} | {
        f"r@{k}": round(float(r), 4) for k, r in enumerate(metrics.recall, start=1)
    }
    (Path(args.model_dir) / f"metrics-{args.split}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def cmd_demo_artifact(args: argparse.Namespace) -> None:
    from .demo import write_demo_artifact

    destination = write_demo_artifact(args.output)
    print(f"wrote demo artifact to {destination} (illustrative predictions only)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tagpuncher", description=__doc__)
    parser.add_argument("--data-dir", default=str(Paths().root))
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="fetch Wikipedia shards from Kaggle")
    download.add_argument("--n-shards", type=int, default=N_SHARDS)
    download.set_defaults(func=cmd_download)

    prepare = subparsers.add_parser("prepare", help="build vocabulary, splits and PECOS files")
    prepare.add_argument("--n-shards", type=int, default=N_SHARDS)
    prepare.add_argument("--n-train", type=int, default=100_000)
    prepare.add_argument("--n-eval", type=int, default=1_000)
    prepare.set_defaults(func=cmd_prepare)

    train = subparsers.add_parser("train", help="train and save a versioned artifact")
    train.add_argument("--n-shards", type=int, default=N_SHARDS)
    train.add_argument("--nr-splits", type=int, default=8)
    train.add_argument("--version")
    train.add_argument("--output")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="score a split and write metrics.json")
    evaluate.add_argument("model_dir")
    evaluate.add_argument("--split", default="testing", choices=["testing", "validation"])
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.set_defaults(func=cmd_evaluate)

    demo = subparsers.add_parser(
        "demo-artifact", help="write a tiny placeholder model for wiring up the stack"
    )
    demo.add_argument("--output", default=str(Paths().artifacts / "demo"))
    demo.set_defaults(func=cmd_demo_artifact)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
