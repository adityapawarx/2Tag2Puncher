FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TAGPUNCHER_MODEL_DIR=/models/current

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tagpuncher ./tagpuncher
RUN pip install --no-cache-dir ".[serve]"

# The artifact is mounted or synced from object storage at boot rather than
# baked in, so model releases are decoupled from image releases.
VOLUME ["/models"]
EXPOSE 8000

CMD ["uvicorn", "tagpuncher.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
