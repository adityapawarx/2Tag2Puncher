# Running the application

TagPuncher is split into a Python package (pipeline + model + API) and a Next.js
demo UI. Both run against a **model artifact**: a directory containing the PECOS
preprocessor, the XR-Linear model, the tag list and a manifest.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve,dev]"
```

## 2. Get an artifact

Either train one from the Wikipedia corpus (hours, tens of GB of downloads):

```bash
tagpuncher download                 # 37 JSONL shards from Kaggle
tagpuncher prepare                  # vocabulary, splits, PECOS files
tagpuncher train --version 2026-08-21
tagpuncher evaluate data/artifacts/2026-08-21 --split testing
```

…or write the bundled placeholder, which trains in a second and exists purely to
exercise the API and UI:

```bash
tagpuncher demo-artifact --output data/artifacts/demo
```

The placeholder knows six topics and its predictions are illustrative only.

## 3. Serve

```bash
TAGPUNCHER_MODEL_DIR=data/artifacts/demo uvicorn tagpuncher.serve.api:app --reload
```

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/healthz` | model version and label count |
| `POST /v1/tag` | `{text, top_k, min_score}` -> ranked tags |
| `POST /v1/tag/document` | multipart PDF/text upload |

```bash
curl -s localhost:8000/v1/tag -H 'content-type: application/json' \
  -d '{"text": "an inverted index maps terms to documents", "top_k": 3}'
```

Each tag carries the indices of the chunks that produced it, so a client can show
which passage a tag came from.

## 4. Demo UI

```bash
cd web && npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Deployment notes

- The artifact is loaded once per process in the FastAPI `lifespan` hook; loading
  costs seconds, inference costs milliseconds, and the whole thing is CPU-only.
- `Dockerfile` builds the API without the artifact baked in — mount it at
  `/models/current` or sync it from object storage on boot so model releases are
  independent of image releases.
- Memory, not CPU, is the binding constraint when picking a host: the resident
  model dominates.
