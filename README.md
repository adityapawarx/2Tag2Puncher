# 2Tag2Puncher

TagPuncher automatically generates relevant tags for a piece of text, addressing the
**Reverse Folksonomy** problem: instead of asking people to tag content, it learns tags
that people have *already* implicitly assigned.

The key insight is that a Wikipedia article section is, in effect, hand-tagged: every
outgoing wiki-link in it names a concept a human editor judged relevant to that passage.
Treating those link targets as labels yields a large, free, human-curated training set.
TagPuncher trains an extreme multi-label classification (XMC) model on ~100k Wikipedia
articles and can then tag arbitrary text — including documents that carry few or no tags
of their own.

## Table of Contents

* [Problem Statement](#problem-statement)
* [Approach](#approach)
* [Repository Contents](#repository-contents)
* [Pipeline](#pipeline)
* [Model](#model)
* [Results](#results)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Known Limitations](#known-limitations)

## Problem Statement

Traditional folksonomies rely on manual tagging and suffer from:

* **Synonymy:** different terms used to describe the same concept.
* **Polysemy:** the same term carrying multiple meanings.
* **Sparsity:** many items receive few or no tags at all.

These limitations degrade retrieval and content management in large digital libraries and
academic repositories.

## Approach

TagPuncher combines statistical and semantic signals:

* **Statistical** — TF-IDF over word unigrams and bigrams represents the input text.
* **Semantic** — labels are embedded via PIFA (the mean TF-IDF vector of the documents
  that carry them) and recursively clustered into a label tree, so semantically related
  tags share structure and rare tags borrow strength from their neighbours.

The label space contains tens of thousands of Wikipedia titles, which rules out a flat
one-vs-rest classifier. The hierarchical label tree of
[PECOS / XR-Linear](https://github.com/amzn/pecos) makes training and inference tractable
at that scale.

## Repository Contents

| File | Description |
| --- | --- |
| `main.ipynb` | The full implementation: data acquisition, preprocessing, training, evaluation, and an interactive "tag your own text" section. |
| `main_ipynb.pdf` | Static PDF export of the notebook, including saved outputs. |
| `TagPuncher_Reseach_Paper_on_Reverse_Folksonomy.pdf` | Research paper describing the problem, method, and findings. |
| `TagPuncher_PPT1_Project_Proposal.pdf` | Initial project proposal deck. |
| `2Tag2Puncher_PPT2_Update.pdf` | Project update deck. |

## Pipeline

`main.ipynb` runs top to bottom in six stages.

### 1. Data acquisition

Downloads 37 JSONL shards (`enwiki_namespace_0_{1..37}.jsonl`) of the Kaggle dataset
[`wikimedia-foundation/wikipedia-structured-contents`](https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents)
into `data/`. Each line is one structured article: `identifier`, `name`, `url`, a nested
`sections` tree (`has_parts`), and `infoboxes`.

### 2. Label vocabulary

Streams every shard once, walking the section tree to collect all `links[].url` and
`images[].content_url`, and writes:

* `all_links_summary.csv` — per article: link count, word count, and
  `link_density = links / words`.
* `unique_links_counts.csv` — corpus-wide frequency of each linked URL.

URLs are then mapped to article titles and filtered down to a candidate tag vocabulary
(`links.csv`) by dropping asset links (`.svg`, `.png`, `.pdf`, `.html`, …), links whose
title contains "wiki" (meta pages), and any link appearing fewer than 50 times.

A Zipf rank/frequency log-log plot with a linear fit confirms the label distribution
follows the expected power law — the motivation for using XMC.

### 3. Document selection and splits

Articles are filtered to good training documents: navigational pages are dropped
(titles starting with `index`, `list`, `alphabetical`, `timeline`, `iucn`, `redirect`),
requiring more than 25 links and more than 100 words, deduplicated by URL and by
identifier (keeping the highest link density). Sorted by `link_density` descending:

| Split | Selection | Size |
| --- | --- | --- |
| Training | highest link density | 100,000 pages |
| Testing | next slice | 1,000 pages |
| Validation | **lowest** link density | 1,000 pages |

The validation split is deliberately the sparsely-linked tail — content "with little to no
tags", which is the scenario TagPuncher is built for.

### 4. Example construction

A second pass flattens each article's section tree into paragraph and list-item leaves,
keeping a breadcrumb section path. Consecutive leaves are greedily merged while the
combined length stays at or below 450 words, unioning their label sets, so each training
example is a chunk long enough for TF-IDF to be informative.

Text is lowercased, stripped of punctuation, whitespace-normalised, and accent-folded
(NFKD). Labels are intersected with the vocabulary and factorised into integer IDs, then
written in PECOS's expected format:

```
1,7,42<TAB>cleaned chunk text on a single line
```

producing `training-data.txt`, `testing-data.txt`, `validation-data.txt`, and
`output-labels.txt` (one tag name per line, ordered by label ID).

### 5. Training

See [Model](#model).

### 6. Evaluation and inference

Predictions are produced in batches and scored with `smat_util.Metrics.generate(topk=10)`.
The notebook also prints sampled chunks alongside their predicted tags and scores, repeats
the exercise on the low-density validation split, and ends with a cell where arbitrary text
can be pasted in for tagging.

## Model

`CustomPECOS` (adapted from the PECOS reference wrapper) bundles a preprocessor, an
XR-Linear model, and the tag name list.

1. **Featurisation** — TF-IDF with `min_df=40`, `max_features=100000`, word 1–2 grams,
   `max_df_ratio=0.90`, sublinear TF, smoothed IDF, L2 normalisation, English stop words.
2. **Label embeddings** — `LabelEmbeddingFactory.create(Y, X, method="pifa")`.
3. **Label tree** — `Indexer.gen(label_feat, nr_splits=8)`, recursive balanced 8-way
   k-means over the label embeddings.
4. **Classifier** — `XLinearModel.train(X, Y, C=cluster_chain,
   negative_sampling_scheme="tfn", threshold=0.1)`, a hierarchy of one-vs-rest linear
   classifiers trained with teacher-forced negatives and weight sparsification.

`save()` / `load()` persist the preprocessor, the XR-Linear model, and `output_items.json`
under a model folder, so a trained model can be reloaded for inference without retraining.

## Results

Precision and recall at *k* on the held-out test split:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Precision | 86.65 | 84.35 | 82.11 | 80.13 | 78.18 | 76.58 | 74.69 | 72.86 | 71.08 | 69.27 |
| Recall | 5.32 | 10.11 | 14.47 | 18.55 | 22.32 | 25.92 | 29.14 | 32.10 | 34.84 | 37.27 |

Precision stays high deep into the ranking, which is what matters for tag suggestion:
the top handful of proposed tags are almost always relevant. Recall is low by
construction — densely-linked source articles carry far more than 10 gold tags each.

## Getting Started

Read `TagPuncher_Reseach_Paper_on_Reverse_Folksonomy.pdf` first for the motivation and
method, then work through `main.ipynb`.

The notebook was developed in Google Colab and expects a GPU-less but high-memory runtime
with substantial disk space (the raw shards are tens of gigabytes).

```bash
pip install scikit-learn scipy==1.9.3 matplotlib orjson kagglehub libpecos pandas numpy tqdm joblib
```

Kaggle credentials are required for `kagglehub` to download the dataset.

## Usage

1. Run the acquisition cells to download the shards into `data/`.
2. Run the preprocessing cells to produce `links.csv`, the per-split processed CSVs, and
   the PECOS-format `.txt` files.
3. Train with `CustomPECOS.train("training-data.txt", "output-labels.txt")` and save the
   model.
4. Evaluate on `testing-data.txt` and `validation-data.txt`.
5. To tag your own text, load a saved model and pass a list of strings:

```python
model = CustomPECOS.load("./model_moreGrams/pecos-CustomPECOS-model")
Y_pred = model.predict(["your document text here"])

for j in range(Y_pred.indptr[0], Y_pred.indptr[1]):
    print(f"{Y_pred.data[j]:.4f}  {model.output_items[Y_pred.indices[j]]}")
```

## Known Limitations

* Paths in the notebook are a mix of Colab-absolute (`/content/...`) and relative, so some
  cells need editing to run elsewhere.
* Cells are not fully order-independent; run the notebook top to bottom.
* Infobox links are attached to every chunk of an article, adding label noise.
* Sampling for the qualitative checks is unseeded, so those printed examples vary between
  runs.
* There is no `requirements.txt`; dependency versions other than the `scipy` pin required
  by `libpecos` are unpinned.
