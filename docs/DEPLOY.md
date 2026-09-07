# Deployment

## Option A — script (one command)

```bash
pip install -U "huggingface_hub>=1.0"
hf auth login                      # a WRITE token
./deploy.sh <your-username> ai-media-pipeline
```

The script validates `config.json`, runs the test suite, creates the Space on
ZeroGPU (`--flavor zero-a10g`), forwards your token as the `HF_TOKEN` secret and
uploads everything except caches and generated output.

Deploying to a different tier:

```bash
./deploy.sh <username> ai-media-pipeline l40sx1     # dedicated 48 GB GPU
./deploy.sh <username> ai-media-pipeline cpu-basic  # UI only, no generation
```

`--flavor` accepts: `cpu-basic`, `cpu-upgrade`, `zero-a10g` (ZeroGPU),
`t4-small`, `t4-medium`, `l4x1`, `l4x4`, `l40sx1`, `l40sx4`, `l40sx8`,
`a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`, `a100-large`,
`a100x4`, `a100x8`.

> Hardware and secret flags are **ignored when the Space already exists**. For an
> existing Space, change them under Settings.

## Option B — git

```bash
git clone https://huggingface.co/spaces/<username>/ai-media-pipeline
cd ai-media-pipeline
cp -r /path/to/this/repo/* .
git add -A && git commit -m "Deploy AI Media Pipeline" && git push
```

Then set hardware and the `HF_TOKEN` secret under Settings.

## Post-deploy checklist

1. **Hardware** — Settings → Hardware. ZeroGPU needs a PRO account to host
   (free accounts in good standing may host 2 ZeroGPU Spaces).
2. **`HF_TOKEN`** — Settings → Variables and secrets. Only useful if the
   account behind the token has accepted the gated model's licence.
3. **Build** — the container build takes a few minutes. The app itself starts
   fast because no model is loaded until a tab is used.
4. **Smoke tests** — below.

## Smoke tests

Run them in this order; each first run includes a large download.

**Image** — `a rain-slicked neon alley at night, shallow depth of field`,
26 steps, guidance 4.0, 1024×1024. Expect a PNG and a run-info line naming the
seed and elapsed time.

**Video** — `slow dolly shot across a misty pine forest at dawn`, 65 frames,
24 fps, 30 steps. Expect an MP4. Note that the frame count snaps to 8n+1 and
the resolution to multiples of 32 — the run info reports what was actually
used.

**Text** — `Summarise the trade-offs between flow matching and DDPM sampling.`,
512 tokens, temperature 0.7. If the run info says `(fallback)`, the gated model
was unreachable; check `HF_TOKEN` and the licence.

**Multimodal text** — attach an image and ask `What is in this picture?`. A
grounded answer confirms the processor path works; a generic one means a
text-only model is loaded.

## Operating notes

### Quota

ZeroGPU quota is daily, per account: 5 min free, 40 min PRO, then credits at
$1/10 min. Requested duration — not actual runtime — is what queues you, so
leaving `gpu_seconds_*` far above reality costs priority.

### Disk

Three model families total ~85 GB against a modest ephemeral disk. The manager
evicts other cached snapshots before a download and logs what it removed, so
switching tabs may trigger a re-download. To avoid that entirely, disable the
modules you do not need:

```json
"video": { "enabled": false }
```

Their tab becomes a notice and their footprint disappears.

### Restarts

The disk is ephemeral: a restart clears downloaded models, generated media and
the run log. That is a property of Spaces, not of this app.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails resolving `torch` | The pin is outside what the Space's Python supports | Keep `python_version: "3.12.12"`; ZeroGPU supports 3.10.13 and 3.12.12 only |
| `ChromaPipeline` not found | `diffusers` older than 0.35 | Reinstall from `requirements.txt` |
| 401 / "gated" in the logs | Licence not accepted, or no `HF_TOKEN` | Accept on the model page, add the secret, restart |
| `No space left on device` | Disk guard could not free enough | Disable a module, or raise `system.disk_headroom_gb` |
| "GPU allocation expired" | Render exceeded the requested duration | Fewer steps/frames, or raise `gpu_seconds_*` |
| Black image, no error | fp16 on a T5 encoder | `"dtype": "bfloat16"` |
| `sentencepiece` / tokenizer error | Dependency dropped from the install | Reinstall from `requirements.txt` |
| Video plays but is corrupt | ffmpeg missing, OpenCV fallback used | Ensure `imageio[ffmpeg]` installed; check logs for the fallback warning |
| Uploaded image seems ignored | A text-only fallback model is loaded | The run info says so; fix `HF_TOKEN` or set an open VL model as primary |
