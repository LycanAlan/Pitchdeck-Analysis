# Multi-stage: the compiler toolchain that some wheels need never reaches the
# runtime image, and the encoder weights are baked in at build time so a cold
# container start is not a several-hundred-megabyte model download.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first: sentence-transformers would otherwise pull the CUDA
# build and add roughly 2 GB of unusable GPU libraries to the image.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copied before the download step so the model names come from the one place
# they are declared, and so editing application code does not re-download them.
COPY pitchlens/ ./pitchlens/
RUN python -c "from pitchlens.config import settings; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(settings.models.embedding); \
CrossEncoder(settings.models.cross_encoder)"

COPY api/ ./api/
COPY scripts/ ./scripts/
COPY app.py ./

# The ingested corpus ships with the image. Without it a deployed container has
# nothing to answer from, and re-ingesting on boot would cost hundreds of vision
# calls. The FAISS index is deliberately NOT copied — it rebuilds from these
# documents in ~4s at startup, which avoids serving a stale index.
COPY data/documents/ ./data/documents/
COPY data/eval/ ./data/eval/

RUN useradd --create-home --uid 1000 pitchlens \
 && mkdir -p /app/data/indices /app/data/decks /app/results \
 && chown -R pitchlens:pitchlens /app /opt/models
USER pitchlens

# Render, Fly and Cloud Run all inject the port to bind. Default to 8000 for
# local runs and docker-compose.
ENV PORT=8000
EXPOSE 8000 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ['PORT']}/health\")"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
