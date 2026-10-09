# AlphaAI production inference server.
#
# Build (from the repository root, so the build context includes the package):
#   docker build -f deploy/Dockerfile -t alphaai-api .
#
# Run (mount a volume so the weights survive restarts):
#   docker run -d --name alphaai -p 8090:8090 \
#     -v alphaai-data:/data \
#     -e ALPHAI_CORS_ORIGINS=https://your-app.vercel.app \
#     -e ALPHAI_INFERENCE_TOKEN=... \
#     alphaai-api
#
# Conversation history is optional: set DATABASE_URL (Supabase) to enable it. A
# long-lived host like this one should use the direct connection (port 5432) or
# Supavisor's *session* mode, not the transaction pooler. See docs/DATABASE.md.
#
# The GGUF weights are downloaded on first boot into /data/models (about
# 0.5 GB) and are never baked into the image or committed to git.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ALPHAI_HOST=0.0.0.0 \
    ALPHAI_MODELS_DIR=/data/models \
    ALPHAI_STATE_DIR=/data/state \
    ALPHAI_LOG_DIR=/data/logs \
    ALPHAI_WORKSPACE_DIR=/data/workspace \
    ALPHAI_LOG_LEVEL=info

WORKDIR /app

# Install the project through its own packaging configuration. The published
# CPU wheel index provides prebuilt llama-cpp-python wheels, so a CPU-only host
# never has to compile ggml from source.
COPY pyproject.toml README.md /app/
COPY alphaai /app/alphaai
COPY configs /app/configs
# The SQL migrations ship with the image so `alphaai db migrate` (or
# ALPHAI_DB_MIGRATE_ON_START=1) can apply the schema from inside the container.
COPY supabase /app/supabase
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        ".[api,llama,db]"

COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh

# Model weights, memory and logs live here and must be a persistent volume.
VOLUME ["/data"]
EXPOSE 8090

CMD ["sh", "/app/deploy/entrypoint.sh"]
