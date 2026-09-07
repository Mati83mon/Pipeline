# AI Media Pipeline — container for local GPU boxes, RunPod, Vast.ai and the
# like. The Hugging Face Space itself uses the Gradio SDK and ignores this file.
#
# The PyPI torch wheels bundle their own CUDA runtime, so a slim Python base
# plus the host NVIDIA driver is enough — no multi-gigabyte CUDA devel image.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # HF_HOME replaces the deprecated TRANSFORMERS_CACHE / DIFFUSERS_CACHE vars.
    HF_HOME=/home/app/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ffmpeg for video encode/decode; libgl+libglib for OpenCV's fallback path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged, with the UID Hugging Face containers expect.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
RUN mkdir -p /app/outputs "$HF_HOME" && chown -R app:app /app "$HF_HOME"

USER app
EXPOSE 7860

# Gradio answers /config once the server is up; / can 200 from a proxy before
# the app is actually ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/config >/dev/null || exit 1

CMD ["python", "app.py"]
