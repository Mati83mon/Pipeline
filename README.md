---
title: AI Media Pipeline
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Image, video and vision-language generation in one Space
tags:
  - text-to-image
  - text-to-video
  - image-text-to-text
  - zerogpu
  - not-for-all-audiences
models:
  - lodestones/Chroma1-HD
  - Lightricks/LTX-Video
  - orcarouter/Qwen3.8-27B-Uncensored-FP8
  - Qwen/Qwen3-VL-8B-Instruct
---

# AI Media Pipeline

Three generative modules behind one Gradio UI, with exactly one model resident
at a time so they fit inside a single GPU allocation.

| Tab | Model | Pipeline | On disk |
|---|---|---|---|
| **Image** | [`lodestones/Chroma1-HD`](https://huggingface.co/lodestones/Chroma1-HD) | `ChromaPipeline` | ~28 GB |
| **Video** | [`Lightricks/LTX-Video`](https://huggingface.co/Lightricks/LTX-Video) | `LTXPipeline` / `LTXImageToVideoPipeline` | ~27 GB |
| **Text** | [`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) → fallback [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `AutoModelForImageTextToText` | ~31 GB / ~18 GB |

> This Space is configured to run uncensored models. Everything it produces is
> the operator's responsibility, under the licences of the individual models and
> the Hugging Face terms of service.

## How it runs

The app targets **ZeroGPU**, where a GPU is attached per request rather than
held for the life of the Space. That shapes three things:

- **Weights live in host RAM; the GPU sees them only during a request.**
  `ModelManager.ensure_loaded()` runs outside the GPU allocation, and the actual
  `pipe(...)` call runs inside a `@spaces.GPU` function which moves the model to
  CUDA and back. The same code path is a no-op off ZeroGPU, so the Space, a
  dedicated GPU Space, Docker and a CPU box all run the identical file.
- **GPU time is requested per call.** `duration` is computed from the request
  (steps, frames, tokens), so a 26-step image asks for less wall clock — and
  gets better queue priority — than a 60-frame video.
- **Disk is the real constraint, not VRAM.** Three model families total ~85 GB
  against a Space's modest ephemeral disk. Before a download the manager checks
  free space and evicts other snapshots from the Hub cache (`disk_guard_enabled`
  in `config.json`). Switching tabs can therefore trigger a re-download; that is
  the trade for keeping all three modules available.

Practical consequence: **the first run of each tab is slow** — tens of GB have
to arrive before anything renders. Later runs on the same tab are fast.

### Quota

ZeroGPU quota is per account, per day: 5 minutes free, 40 minutes with PRO,
then pre-paid credits at $1 per 10 minutes. A video render is minutes of GPU
time, so budget accordingly.

## Setup

### 1. Hardware

Space **Settings → Hardware → ZeroGPU**. The default `large` allocation (48 GB)
covers every module here. To force the full 96 GB card, pass `size="xlarge"` to
`gpu_task` in `app.py` — it costs 2× quota per second.

### 2. `HF_TOKEN` secret (needed for the default text model)

`orcarouter/Qwen3.8-27B-Uncensored-FP8` is **gated**. Without a token the Text
tab silently falls back to `Qwen/Qwen3-VL-8B-Instruct`.

1. Accept the licence on the [model page](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8).
2. Create a token with **read** scope at <https://huggingface.co/settings/tokens>.
3. Space **Settings → Variables and secrets → New secret**, name `HF_TOKEN`.

The token is read from the environment at load time and passed to every
`from_pretrained` call. It is never written to the log or the UI.

## Configuration

Everything tunable lives in `config.json`, validated at startup — a typo raises
a clear `ConfigError` instead of a `KeyError` at click time.

```jsonc
{
  "image": { "enabled": true, "model_id": "lodestones/Chroma1-HD", "dtype": "bfloat16" },
  "video": { "enabled": true, "model_id": "Lightricks/LTX-Video" },
  "llm":   { "enabled": true, "model_id": "orcarouter/Qwen3.8-27B-Uncensored-FP8",
             "fallback_model_id": "Qwen/Qwen3-VL-8B-Instruct", "load_in_4bit": false },
  "system": { "disk_guard_enabled": true, "disk_headroom_gb": 12, "keep_outputs": 40 }
}
```

Useful edits:

- **Set `"enabled": false`** on a module you do not need. Its tab becomes a
  notice and its ~30 GB never competes for disk.
- **`dtype`** defaults to `bfloat16`. Do not switch the diffusion models to
  `fp16`: their T5 text encoders overflow to NaN, which shows up as black
  output rather than as an error.
- **`load_in_4bit: true`** shrinks the 27B text model at some quality cost.
- **`gpu_seconds_*`** feed the ZeroGPU duration estimate. Raise them if runs get
  cut off mid-render, lower them for better queue priority.

## Local and Docker

```bash
pip install -r requirements.txt
python app.py                      # http://localhost:7860
```

```bash
docker build -t ai-media-pipeline .
docker run --gpus all -p 7860:7860 -e HF_TOKEN=$HF_TOKEN \
  -v "$PWD/hf-cache:/home/app/.cache/huggingface" ai-media-pipeline
```

Mount the cache volume — without it every container start re-downloads tens of
gigabytes.

Off ZeroGPU the `@spaces.GPU` decorator is transparent and the app uses whatever
CUDA device is present, or CPU. On CPU the UI works and the text model is usable
with small token budgets; image and video generation are not practical.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q          # 72 tests, no torch or GPU required
```

They cover config validation, LTX frame/resolution alignment, seed handling,
log rotation, frame normalisation and the full UI build.

## Layout

```
app.py                  Gradio entry point, handlers, @spaces.GPU boundaries
ui_components.py        Tab builders (config passed in, never imported globally)
config.json             Models, limits, system policy
pipeline/
├── config.py           Schema validation
├── runtime.py          ZeroGPU shim, dtype/VRAM probing, disk budgeting
├── models.py           ModelManager: load, evict, generate
├── media.py            Video encode/decode, output pruning
└── history.py          Bounded JSONL run log
tests/test_pipeline.py  Test suite
docs/AUDIT.md           Every defect fixed relative to the first draft
docs/DEPLOY.md          Deployment and troubleshooting
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Text tab says `(fallback)` | The gated 27B model is unreachable. Accept its licence and set `HF_TOKEN`. |
| `No space left on device` | Disk guard could not free enough. Disable a module in `config.json` or raise `disk_headroom_gb`. |
| "GPU allocation expired" | The render needed more than the requested duration. Cut steps or frames, or raise `gpu_seconds_*`. |
| Black or empty image | Almost always `fp16` on a T5 encoder. Set `dtype` back to `bfloat16`. |
| First run takes many minutes | Expected: the model is downloading. Watch the Space logs. |
| `ChromaPipeline` not found | `diffusers` is older than 0.35. Reinstall from `requirements.txt`. |

## Licence

Code: Apache-2.0 (`LICENSE`). Each model carries its own licence, which governs
what you may do with its output.
