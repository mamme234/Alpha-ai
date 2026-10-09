#!/bin/sh
# AlphaAI production entrypoint.
#
# 1. Install the configured lightweight model if it is not already on the
#    persistent volume (never re-downloads on restart).
# 2. Start the existing FastAPI server with `alphaai serve`.
#
# The model weights live on the mounted volume; the API reports the real model
# state through /api/health, so a failed install is visible rather than hidden.
set -eu

MODEL_ID="${ALPHAI_MODEL_ID:-qwen2.5-0.5b-instruct-gguf}"
MODELS_DIR="${ALPHAI_MODELS_DIR:-/data/models}"
# Weights baked into the image by deploy/prefetch_model.py at build time.
CACHE_DIR="${ALPHAI_MODEL_CACHE_DIR:-/opt/model-cache}"

# An empty ALPHAI_CORS_ORIGINS resolves to an empty allow-list, which blocks
# every browser request. Warn instead of failing silently.
if [ -z "${ALPHAI_CORS_ORIGINS:-}" ]; then
  echo "AlphaAI: warning - ALPHAI_CORS_ORIGINS is not set; browsers will only be"
  echo "AlphaAI:           allowed by the configs/alphaai.toml default. Set it to"
  echo "AlphaAI:           your deployed frontend origin, e.g. https://app.vercel.app"
fi

if [ "${ALPHAI_SKIP_MODEL_INSTALL:-0}" != "1" ]; then
  # First boot: seed the persistent disk from the weights the image already
  # carries (deploy/prefetch_model.py fetched them at build time), so the port
  # opens in seconds instead of after a 491 MB transfer. The installer then
  # verifies the copy (size + SHA-256) and load-tests it — the model is never
  # reported available just because files exist.
  SEEDED=0
  if [ ! -d "${MODELS_DIR}/${MODEL_ID}" ] && [ -d "${CACHE_DIR}/${MODEL_ID}" ]; then
    echo "AlphaAI: seeding ${MODEL_ID} from the image cache onto ${MODELS_DIR}"
    mkdir -p "${MODELS_DIR}/${MODEL_ID}"
    cp -r "${CACHE_DIR}/${MODEL_ID}/." "${MODELS_DIR}/${MODEL_ID}/"
    SEEDED=1
  fi

  if [ -d "${MODELS_DIR}/${MODEL_ID}" ] && [ "${SEEDED}" != "1" ]; then
    echo "AlphaAI: model ${MODEL_ID} already present in ${MODELS_DIR}"
  else
    echo "AlphaAI: installing ${MODEL_ID} into ${MODELS_DIR}"
    python -m alphaai.cli models install "${MODEL_ID}" || \
      echo "AlphaAI: model install failed; /api/health will report it as unavailable"
  fi
fi

exec python -m alphaai.cli serve
