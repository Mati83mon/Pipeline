# Audit of the original draft

The first version of this project was generated as a "copy these ten files
verbatim" manifest. This document records every defect found while rewriting it,
so nobody reintroduces one by pasting the old code back.

Findings are grouped by whether they stopped the app from working at all, broke
a feature quietly, or were correctness/robustness problems.

---

## A. Blockers — the Space could not have started or served a request

### A1. No YAML front matter in `README.md`

A Gradio Space is configured by the YAML block at the top of `README.md`
(`sdk`, `sdk_version`, `app_file`, `python_version`). The draft's README began
with a Markdown heading, so the Space had no SDK declaration and would not
build.

**Fixed:** full front matter, with `sdk_version` matching the pinned Gradio in
`requirements.txt`.

### A2. No ZeroGPU integration

There was no `import spaces` and no `@spaces.GPU`. On ZeroGPU hardware a GPU is
only attached inside a decorated function; outside one, PyTorch runs in an
emulation mode. Every generation would have failed or fallen back to CPU.

**Fixed:** `pipeline/runtime.py` provides a `gpu_task` decorator that maps to
`spaces.GPU` when available and is transparent otherwise, so one file runs on
ZeroGPU, on a dedicated GPU, in Docker and on CPU.

### A3. Wrong pipeline class for the image model

`config.json` declared `FluxPipeline` and `utils.py` imported it. But
`lodestones/Chroma1-HD` ships `model_index.json` with
`"_class_name": "ChromaPipeline"` and a `ChromaTransformer2DModel`. Chroma is
FLUX-*derived*, not FLUX; `FluxPipeline.from_pretrained` raises on the
transformer config.

**Fixed:** `ChromaPipeline`, driven by `pipeline_class` in config and resolved
dynamically with a clear error if the installed `diffusers` is too old.

### A4. `safety_checker` arguments passed to a FLUX-family pipeline

```python
FluxPipeline.from_pretrained(model_id, safety_checker=None,
                             requires_safety_checker=False)
```

Those parameters belong to `StableDiffusionPipeline`. FLUX and Chroma have no
safety checker component at all, so the arguments are at best ignored and at
worst rejected. They also created the false impression that a filter was being
switched off.

**Fixed:** removed. These pipelines ship no safety module.

### A5. The video model is not loadable by `diffusers`

`ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8` is a **ComfyUI** repository:
GGUF quantisations plus split `diffusion_models/`, `text_encoders/` and `vae/`
safetensors, a Gemma-3 text encoder and ComfyUI workflow JSON. It has no
`model_index.json`. Neither `from_pretrained` nor `from_single_file` can load
it, so the "primary" video model could never have worked.

**Fixed:** the video module uses `Lightricks/LTX-Video`, which ships a real
diffusers layout. The ComfyUI repo is retained in `config.json` under
`comfyui_only_reference` with `diffusers_compatible: false` so the choice is
documented rather than silently dropped.

### A6. `LTXImageToVideoPipeline` used for text-to-video

The draft loaded the image-to-video class and then called it without an image
whenever the user left the upload empty. That pipeline requires `image`.

**Fixed:** `LTXPipeline` for text-to-video, `LTXImageToVideoPipeline` for
image-to-video, constructed from the already-loaded components
(`i2v_cls(**pipe.components)`) so mode switching costs no extra VRAM and no
extra download.

### A7. `LTXImageToVideoPipeline.from_single_file` does not exist

The second load attempt called a method the LTX pipelines do not implement
(`FromSingleFileMixin` is not mixed in), so the "fallback" path raised
`AttributeError` before it could reach the real fallback. It also picked
`safetensors[0]` — the first filename in Hub order, which in that repo is a VAE
or text encoder shard, not the transformer.

**Fixed:** removed; the load path no longer guesses at checkpoint files.

### A8. Missing `sentencepiece`

Both Chroma and LTX use a sentencepiece-backed `T5Tokenizer` (`spiece.model`).
`requirements.txt` listed neither `sentencepiece` nor `protobuf`, so tokenizer
construction fails on a clean Space image.

**Fixed:** both added.

---

## B. Silent failures — features that appeared to work but did not

### B1. `gr.Error` called instead of raised

```python
except Exception as exc:
    gr.Error(f"Image generation failed: {exc}")   # constructs, discards
    return None, f"❌ Error: {exc}", *refresh_status()
```

`gr.Error` is an exception class. Constructing one has no effect; the user saw
no error toast. The same bug appeared in all three handlers.

**Fixed:** `raise gr.Error(...)`, with backend exceptions translated into
actionable messages (gated repo, out of memory, expired allocation, full disk).

### B2. Multimodal input was never sent to the model

```python
if image is not None:
    content_parts.append("<image>")
...
inputs = tokenizer(text, return_tensors="pt")
```

The literal strings `<image>` and `<video>` were concatenated into the prompt
and the pixels were dropped. The model received text describing that an image
existed and nothing more. Uploading a picture changed the output only by
inserting a stray token.

**Fixed:** `AutoProcessor` with a proper chat template carrying real image
content; video is decoded into evenly spaced frames first. When the loaded
model is text-only, the UI says the media was ignored instead of pretending.

### B3. Wrong auto class for the text model

`orcarouter/Qwen3.8-27B-Uncensored-FP8` is `Qwen3_5ForConditionalGeneration`,
task `image-text-to-text`, with a `Qwen3VLProcessor`. The draft loaded it with
`AutoModelForCausalLM`, which does not map that architecture.

**Fixed:** `AutoModelForImageTextToText` first, `AutoModelForCausalLM` as the
fallback for genuinely text-only models.

### B4. `extract_video_frames` was dead code

The helper existed, was documented in the UI ("frames extracted for analysis"),
and was never called anywhere.

**Fixed:** wired into the text path, bounded by `llm.max_video_frames`.

### B5. `fp16` on T5 text encoders

`config.json` set `"dtype": "fp16"` for both diffusion models. T5 encoders
overflow to NaN in fp16 — the classic symptom is a black image with no error.

**Fixed:** `bfloat16` default, with `resolve_dtype` degrading sensibly when
bf16 is unavailable, and the hazard documented in the README.

### B6. The fallback flag was never set for images

The History tab read `params.fallback`, but only the video and text paths wrote
it. The image row always displayed `NO`, whether or not a fallback was used.

**Fixed:** `fallback` is a first-class log field written by the manager for
every module, from the actually-loaded model.

### B7. "Reload config" only half-reloaded

`ui_components.py` did `CONFIG = load_config("config.json")` at import time.
The button rebuilt the `ModelManager`, so the backend picked up new limits while
every slider kept the bounds captured at process start — the two could disagree
without any warning.

**Fixed:** config is passed into every builder, and the button is gone. Gradio
cannot rebuild the component tree of a running app, so the honest control is a
Space restart; the System panel offers "Unload model", which is a thing the app
can actually do.

### B8. `HF_TOKEN` was never used

The README told users to add the secret for gated models; no code read it. The
gated 27B model would have failed with a 401 that the error message blamed on
something else.

**Fixed:** `runtime.hf_token()` reads it and every `from_pretrained` receives
it; a 401 is translated into "accept the licence and set `HF_TOKEN`".

---

## C. Correctness and robustness

### C1. Disk exhaustion was never considered

The three model families total roughly 85 GB against a Space's modest ephemeral
disk. The draft downloaded into `./model_cache` with no accounting, so the
second module downloaded would fill the disk and kill the Space with an opaque
`OSError`.

**Fixed:** `ensure_disk_budget` checks free space before each download and
evicts least-recently-used snapshots from the Hub cache, reporting what it
removed.

### C2. Unloading did not free VRAM deterministically

`del self.models[target]` drops one reference. Pipelines hold references to
their own submodules and the local `pipe` in the caller frame may still be
alive, so the weights could survive the "unload" and the next load would OOM.

**Fixed:** components are moved to CPU first, then dropped, then `free_vram()`.

### C3. `torch.cuda.get_device_properties(0)` unguarded

Called whenever the status box refreshed. On a CPU box, or before a device
exists, it raises — and it ran on `demo.load`, i.e. on every page open.

**Fixed:** every CUDA probe is guarded and degrades to a descriptive string.

### C4. `int(seed)` on a cleared Number field

`gr.Number` yields `None` when the user clears it; `int(None)` raises
`TypeError` before any generation starts. `seed != -1` also mis-handled
`-1.0` from float-valued Numbers.

**Fixed:** an explicit "Randomise seed" checkbox plus `_resolve_seed`, which
tolerates `None`, floats, junk and out-of-range values.

### C5. Division by zero in the progress callback

```python
progress(0.1 + 0.85 * (step_index + 1) / steps, ...)
```

`steps` came straight from the slider. The config allowed `min_steps: 1`, but
nothing stopped a programmatic call with `0`.

**Fixed:** step counts are validated against config bounds, and per-step
progress no longer crosses the GPU-task boundary (see C10).

### C6. `temperature=0` was passed with `do_sample=False`

Recent `transformers` releases reject generation config that combines greedy
decoding with sampling parameters — it is an error now, not a warning.

**Fixed:** `temperature` and `top_p` are only sent when sampling is on.

### C7. Truncated video files on encoder failure

`save_video_frames` wrapped the whole imageio loop in one `try`. A failure
mid-loop left the writer unclosed and a partial `.mp4` on disk, then the OpenCV
fallback wrote to the same path. The fallback also indexed `frames[0]` without
checking the list was non-empty.

**Fixed:** `finally: writer.close()`, the partial file is deleted before the
fallback runs, empty input raises `MediaError`, and frames are normalised
(float→uint8, grayscale→RGB, RGBA→RGB) with H.264's even-dimension requirement
enforced.

### C8. Unbounded log and unbounded output directory

`generation_log.jsonl` grew forever and `get_history` read the entire file with
`readlines()` on every refresh. Generated media accumulated on the same disk the
model cache needs. A single malformed line broke the History tab outright.

**Fixed:** bounded tail read, size-capped log with rotation, per-line JSON
errors skipped, `keep_outputs` prunes old media, and writes are mutex-guarded so
concurrent runs cannot interleave partial lines.

### C9. No concurrency guard on the manager

`demo.queue(default_concurrency_limit=1)` limits queued events, but `demo.load`
handlers and system buttons run outside that limit and touched the same manager.

**Fixed:** `ModelManager` guards load/unload with an `RLock`.

### C10. Per-step progress cannot cross a ZeroGPU boundary

The draft passed a `callback_on_step_end` closure that called Gradio's
`progress` from inside the diffusion loop. Under ZeroGPU the GPU function is
executed through the `spaces` runtime, and a live Gradio progress handle is not
something to rely on across that boundary.

**Fixed:** progress is reported around the GPU call in stages (prepare → load →
generate → done) with an ETA derived from the same `gpu_seconds_*` values that
size the allocation. Honest staging beats a progress bar that may not update.

### C11. `deploy.sh` used flags that do not exist

`huggingface-cli repo create --sdk gradio --hardware gpu-t4 --yes` — `repo
create` has no `--hardware` flag, ZeroGPU is not a `--hardware` value at all,
and `huggingface-cli` has been superseded by `hf`. The script also ended in
`git push --force` against `main`.

**Fixed:** rewritten around `hf repo create` / `hf upload` with a preflight
check, no forced push, and hardware left to the Settings UI where it belongs.

### C12. Dockerfile problems

`TRANSFORMERS_CACHE` and `DIFFUSERS_CACHE` are deprecated in favour of
`HF_HOME`; the image ran as root; `opencv-python` (not `-headless`) pulls X11
libraries the container lacks; the CUDA 12.1 devel base was several gigabytes
and older than what Blackwell needs; and the healthcheck probed `/`, which can
answer before Gradio is ready.

**Fixed:** slim Python base (the torch wheels carry their own CUDA runtime),
`HF_HOME`, non-root UID 1000, headless OpenCV, healthcheck against
`/config`.

### C13. LFS rules that would have swallowed generated output

`.gitattributes` routed `*.png`, `*.jpg` and `*.mp4` through Git LFS while
`outputs/` was not gitignored — so every generated file would land in LFS on the
next commit.

**Fixed:** LFS covers weight formats only; generated media is gitignored.

### C14. Config errors surfaced as `KeyError` at click time

Two modules independently did `json.load` with no validation. A missing key
raised deep inside a handler, minutes into a session.

**Fixed:** `pipeline/config.py` validates the schema once at startup and reports
the offending key path.

### C15. No tests and no CI

**Fixed:** 148 tests that need neither torch nor a GPU, covering config
validation, LTX `8n+1` frame alignment and resolution snapping, seed handling,
log rotation and corruption tolerance, frame normalisation, output pruning and
the full UI build.

Several of them close the loop on findings above rather than testing new code:
A1 (front matter present, `app_file` resolves), the `sdk_version` ↔ `gradio`
pin agreement, `python_version` and the `torch` pin being values ZeroGPU
actually supports, and every `config.json` model appearing in the README's
`models:` list. Each was verified to fail when the corresponding value is
broken — a test that cannot fail is not a test.

`.github/workflows/ci.yml` runs the lint and suite on every push, plus a
second job that resolves `requirements.txt` with `pip --dry-run`. The
resolution job is the cheap version of the slowest failure mode this project
has: an unsatisfiable pin set that only surfaces several minutes into a Space
build, in logs that blame the last package pip happened to touch.

---

## D. Claims in the original README that were not true

| Claim | Reality |
|---|---|
| "FP8 requires Hopper/Ada; A100 will fall back" | ZeroGPU now runs RTX Pro 6000 **Blackwell**, which handles FP8. The hardware table was outdated across the board. |
| "Video: LTX-2.3 uncensored, 21B" | That repo is ComfyUI-format and unloadable by this stack (A5). |
| "27B vision-language LLM" | Correct about the model, but the code loaded it as a causal LM and never passed it an image (B2, B3). |
| "CPU fallback — the LLM can degrade to CPU mode" | The fallback set `device_map="cpu"` on a 27B model; on a Space that swaps to death rather than serving. |
| "Progress bars with ETA" | There was no ETA anywhere, and the per-step callback would not survive ZeroGPU (C10). |
| "History tracking, last 10 generations" | True until a malformed line broke the tab (C8). |
| "All `safety_checker` layers are explicitly disabled" | These pipelines have no safety checker to disable (A4). |

---

## E. Found on the running Space (not in the original draft)

These are mine, not the draft's — surfaced once the Space was live on ZeroGPU
with a valid `HF_TOKEN` and the text model still fell back.

### E1. A forced dtype on a checkpoint that carries its own quantisation

`_load_llm` passed `dtype=resolve_dtype("bfloat16")` unconditionally. But
`orcarouter/Qwen3.8-27B-Uncensored-FP8` declares its own scheme in
`config.json`:

```json
"quantization_config": { ..., "weight_block_size": [128, 128] }
```

It is an **offline block-FP8 (E4M3)** build, tagged `block-fp8` and `vllm`,
whose card states it "serves with the identical vLLM kernel path". Handing
such a repo an explicit dtype overrides the policy its weights were produced
under.

**Fixed:** `_declared_quantization()` reads the repo's config first; when a
scheme is declared, dtype is left to the checkpoint and the decision is
logged. Unquantised repos — including the fallback `Qwen3-VL-8B-Instruct` —
still get the explicit dtype. `dtype: "force:bfloat16"` opts back in.

A second consequence of the same finding: bitsandbytes `load_in_4bit` is no
longer stacked on top of already-quantised weights, which is not a supported
combination. The UI says so rather than failing obscurely.

### E2. A silent fallback with no recoverable cause

The fallback path logged `LOGGER.warning("loading %s failed: %s", ...)` — the
exception's `str()`, no traceback — and told the user only
`could not be loaded (ModelLoadError)`. For a chain of loader attempts that
class name is always `ModelLoadError`: true and useless. A gated 401, an
unsupported architecture, an OOM and a dtype conflict were indistinguishable,
so every diagnosis was guesswork.

**Fixed:** `LOGGER.exception` at both levels, so the Space logs carry the full
traceback; `ModelLoadError` names which auto classes were tried; and
`_short_reason()` puts the exception type plus its first line into the UI
notice.

### E3. The Hub's declared auto class was not attempted

The Hub lists this model under `AutoModelForMultimodalLM` while its config
declares `Qwen3_5ForConditionalGeneration`. Only the latter's auto class was
tried.

**Fixed:** both are attempted (guarded by `getattr`, so a `transformers`
without either still works), then `AutoModelForCausalLM`.

> **Not done, deliberately:** `trust_remote_code` stays `false`. The model repo
> contains no `.py` files at all — only config, tokenizer, chat template and
> safetensors — so there is no remote code to trust. Enabling it would accept
> arbitrary-code-execution risk in exchange for nothing.

### E4. The actual blocker: a bug in `transformers` itself

With E1–E3 in place the logs finally named the cause, and it was upstream. All
three auto classes failed identically, before a single weight was read:

```
transformers/quantizers/quantizer_finegrained_fp8.py", line 195, in update_tp_plan
    updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
AttributeError: 'NoneType' object has no attribute 'get'
```

Reading `transformers==5.16.1`, `update_tp_plan` is:

```python
if "Qwen3" in config.__class__.__name__:
    config.base_model_tp_plan = text_plan          # non-empty, unconditionally

impl = getattr(config, "_experts_implementation", None)
layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)
for plan_attr in ("base_model_tp_plan", "base_model_ep_plan"):
    base_plan = getattr(config, plan_attr, None) or {}
    updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
```

`FP8Experts._impl_tp_layer_overrides` holds exactly one key,
`"deepgemm_megamoe"`, while `_experts_implementation` defaults to `None`
(`configuration_utils.py:333`). So `.get(impl)` returns `None`. The guard the
line needs — `or {}` — is missing.

The crash is therefore **unconditional** for any fine-grained FP8 checkpoint
whose config class name contains `Qwen3`: the same function guarantees a
non-empty `base_plan` two lines earlier, so the comprehension always runs.
Nothing about this project could have avoided it.

**Worked around:** `_patch_fp8_tp_plan()` registers `None -> {}` in that map
before the FP8 load. That is precisely the "no overrides" behaviour the code
intends — the plan is copied unchanged, compares equal, and is left in place.
It is idempotent, scoped to checkpoints that declare FP8, and becomes a no-op
once upstream adds the guard.

A test reproduces the upstream line directly: it asserts `AttributeError`
before the patch and an unchanged plan after, so if `transformers` fixes this
the workaround's necessity is still documented rather than silently load-bearing.

> Worth reporting upstream — the one-character fix is `.get(impl) or {}`.

### E5. A second transformers bug: unanchored module-skip matching

With E4 in place the model **loaded** — `llm ready in 87.7s`, no fallback, 30.9 GB
of weights resident. Generation then failed, and the load report said why:

```
Qwen3_5ForConditionalGeneration LOAD REPORT
model.language_model.layers.{0...63}.mlp.gate_proj.weight_scale_inv | UNEXPECTED
```

Every FP8 scale for `gate_proj`, in all 64 layers, was rejected as unexpected —
the module had no parameter to receive it. `quantizers_utils.should_convert_module`
decides what stays full precision:

```python
should_not_convert = any(
    re.match(f"{key}\\.", full_name)
    or re.match(f"{key}", full_name)      # anchors the start only
    or full_name.endswith(key)
    for key in patterns
)
```

`re.match` does not anchor the end, so a pattern also matches any longer name
that merely starts with it. This checkpoint lists `...mlp.gate` — the MoE
router — as not-to-be-quantised, and that pattern therefore also swallows
`...mlp.gate_proj`. It was left a plain `nn.Linear` holding FP8 weights with
nowhere to put their scales, which is precisely a model that loads and then
dies in the forward pass.

**Worked around:** `_patch_fp8_module_skip_matching()` replaces the middle
clause with `re.fullmatch`, which is what the function's own docstring
describes ("the pattern matches full_name exactly or via regex"). Verified
against real module names:

| module | upstream | anchored |
|---|---|---|
| `...mlp.gate_proj` | skipped (bug) | **converted** |
| `...mlp.up_proj` | converted | converted |
| `...mlp.gate` (router) | skipped | skipped |
| `lm_head` | skipped | skipped |

The patch probes the installed version first and does nothing when it already
classifies `gate_proj` correctly, so it vanishes on a fixed transformers. It
also reassigns the name inside `integrations.finegrained_fp8`, which imports
the function by value at module scope — patching only the source module would
have been a silent no-op. Scope stops there; the other quantiser integrations
are left untouched.

> Also worth reporting upstream: `re.match(key, name)` should be
> `re.fullmatch(key, name)`.

### E6. `GenerationError` reached the UI with no type and no message

The Text tab showed `Error: 'GenerationError'` — quoted, which is the giveaway:
`str(KeyError("x"))` renders as `"'x'"`. `_explain_generation_failure` returned
a bare `str(exc)`, so for the exception classes that actually surface from a
model's forward pass the type was dropped and the text became unreadable. The
same defect as E2, in the path E2 did not cover.

**Fixed:** every generation failure goes through `_short_reason` (type plus
first line); `KeyError`/`AttributeError`/`IndexError` are additionally named as
a structural checkpoint/architecture mismatch rather than anything about the
user's prompt; and each generation failure is logged with `LOGGER.exception`
tagged with the model id, so the traceback is greppable next to the model that
produced it.

That change did its job — it is what produced the traceback in E7 — but it was
not sufficient on its own to fix what the *user* sees. See E8.

### E7. The real generation blocker: `kernels` was never a dependency

With E4 and E5 in place the load is clean: both patches announce themselves,
no `UNEXPECTED` scales, `llm ready in 90.9s`. Generation still failed, and this
time the traceback reached the bottom:

```
modeling_qwen3_5.py:448, in forward
    mixed_qkv = self.in_proj_qkv(hidden_states)
integrations/finegrained_fp8.py:338, in forward
    return fp8_linear(...)
integrations/finegrained_fp8.py:108, in _load_finegrained_fp8_kernel
    raise ImportError(f"finegrained-fp8 kernel unavailable: {_MISSING_KERNELS_MESSAGE}")
ImportError: finegrained-fp8 kernel unavailable: `kernels` is either not
installed or uses an incompatible version (0.16.0 <= version < 0.17.0)
```

Not a bug this time — a missing dependency, and one that nothing about the
project surfaced. **transformers implements no FP8 matmul of its own.** Every
`FP8Linear.forward` ends in `fp8_linear`, which dispatches to DeepGEMM or to
the Triton `finegrained-fp8` kernel, and *both* are fetched from the Hub
through the `kernels` package:

```python
_HUB_KERNEL_MAPPING = {
    "finegrained-fp8": {"repo_id": "kernels-community/finegrained-fp8", "version": 4},
    "deep-gemm":       {"repo_id": "kernels-community/deep-gemm", "version": 2},
}
```

`requirements.txt` never listed `kernels`, because nothing in the load path
needs it: the checkpoint downloads, quantises and reports ready, all 31 GB of
it, and the gap only opens on the first forward pass — inside the GPU
allocation, after the quota is spent.

**Fixed:** `kernels>=0.16,<0.17` is now a first-class dependency, with the
window taken from transformers' own `KERNELS_MIN_VERSION`/`KERNELS_MAX_VERSION`
rather than restated by hand. Verified before pinning: `kernels` 0.16.1 is the
newest release inside that window, and the kernel repo's `v4` branch — the one
`version: 4` resolves to — exports exactly the three symbols transformers looks
up (`matmul_2d`, `matmul_batched`, `matmul_grouped`); `main` still exports the
older `w8a8_*` names and would have failed the symbol check. The build is pure
Triton (`build/torch-cuda/*.py`, no compiled ABI), so it is not tied to a torch
or CUDA version.

`ModelManager._check_fp8_kernels()` additionally probes for the package at load
time and raises a UI notice when it is missing, so the next person meets this
before spending ninety seconds and a GPU slot on it rather than after.

### E8. Only `gr.Error` survives the ZeroGPU process boundary

E6 built a good message. The UI still said `Error: 'GenerationError'`, and the
log said why:

```
File "spaces/zero/wrappers.py", line 238, in gradio_handler
    raise error("ZeroGPU worker error", res.error_cls)
gradio.exceptions.Error: 'GenerationError'
```

ZeroGPU runs a `@spaces.GPU` function in a **forked worker process** and ships
the outcome back over a queue. Reading `spaces` 0.51.0:

```python
def exception_result(exc: Exception) -> ExceptionResult:
    gradio_error = None
    if isinstance(exc, gr.Error):
        gradio_error = exc
    return ExceptionResult(traceback=..., error_cls=exc.__class__.__name__,
                           gradio_error=gradio_error)
```

and in the parent:

```python
if res.gradio_error is not None:
    raise res.gradio_error                                  # message intact
else:
    raise error("ZeroGPU worker error", res.error_cls)      # class name only
```

So the exception object is transported **only when it is a `gr.Error`**.
Everything else is reduced to `exc.__class__.__name__` — for us, the string
`GenerationError`, which is precisely what the UI printed. Every message the
manager composes for a failure inside a GPU task was being thrown away one
process later, no matter how good it was. This also explains why E6 looked like
it had not worked: it had, in the worker's log, where the user never looks.

**Fixed:** `_gpu_safe` wraps each GPU task *inside* the allocation and converts
anything that escapes into a `gr.Error` there, where the message still exists.
Decorator order matters and is the whole point — `@gpu_task` on the outside,
`@_gpu_safe` beneath it, so the conversion happens in the worker rather than in
the parent that never sees the exception. An exception that is already a
`gr.Error` is re-raised unchanged rather than wrapped twice.

Tested from both ends: that a `GenerationError` raised in a task comes out of
each of the three real `_gpu_*` functions as a `gr.Error` carrying its message
(remove the decorator and three tests fail), and that the converted error
survives a `pickle` round trip — which is the property `spaces` actually relies
on to move it between processes.

---

## F. Found by testing the working model

Everything above was found by trying to make the text model *run*. This section
is different: the model runs, its answers are good, and these are the gaps a
structured test session opened up — recorded so the reasoning behind each fix
survives the session that produced it.

### F1. The chat template defaults `reasoning_effort` to `xhigh`

Six test prompts, four token budgets from 800 to 4096, and on the complex ones
the answer never arrived: the model spent the entire budget inside `<think>`,
often producing a nearly complete, fact-checked draft of the answer in there
before running out of room to write it.

That reads like a quirk of the model. It is configuration. The template:

```jinja
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh.
        Please think carefully through the task, validate key assumptions,
        consider plausible alternatives, and prioritize correctness,
        consistency, and clarity in the final answer.' %}
```

The default is `xhigh`, and at that level the template **injects a system
instruction** telling the model to validate assumptions and consider
alternatives. The observed behaviour — long deliberation, self-fact-checking,
explicit uncertainty about a settlement figure — is the model doing exactly
what it was told, by a line nobody in this project wrote or saw.

`medium` injects nothing. It is the neutral setting, not a degraded one.

**Fixed:** `THINKING_MODES` exposes the four combinations the template actually
supports — `deep` (xhigh), `standard` (medium), `brief` (low), `off`
(`enable_thinking: false`, which makes the template emit a pre-closed
`<think>\n\n</think>` so the answer starts immediately). Default is `standard`.
The values are not free text: an effort outside `('xhigh','medium','low')`
raises inside Jinja, so a test pins the table to that tuple.

The fallback model's template has never heard of either flag, and a Jinja
template may `raise_exception` on anything unexpected, so `_apply_template`
retries without them rather than letting a reasoning preset break a model that
has no reasoning mode.

### F2. Nothing distinguished a finished answer from a truncated one

`generate_text` returned whatever came back. A short answer might be the model
judging that enough had been said, or the model being cut off mid-sentence —
opposite verdicts on its quality, and nothing in the returned text tells them
apart. Every "the answer stops mid-word" observation in the test log had to be
made by eye.

**Fixed:** `_hit_token_ceiling` compares the generated length against the
budget *and* checks whether the last token is an EOS id, gathered from the
generation config, the model config and the tokenizer — chat models routinely
stop on `<|im_end|>` rather than the tokenizer's nominal EOS, and only the
generation config knows that. Both halves are needed: length alone flags a
model that stopped naturally on its last allowed token, a missing EOS alone
flags every early stop on a stop-string.

`_truncation_advice` then names the fix rather than the symptom, because the
two cases have different remedies. Running out *inside* the reasoning block
means spend fewer tokens reasoning; running out while writing the answer means
raise the limit. Telling them apart needs a decode with
`skip_special_tokens=False` — `<think>` and `</think>` are special tokens, so
the user-facing decode drops them.

### F3. The Text tab could not continue its own answer

Fully stateless: every click built `[system, user]` from scratch. Asking it to
"carry on" produced invented continuations, because the model had no access to
what it had written.

**Fixed:** a single `gr.State` holding the last answer, and a "Continue last
answer" button that feeds it back with the original request. Deliberately not a
chat: one model resident at a time is the constraint the whole app is built
around, and a transcript is not what the observed problem needed.

One trap worth recording: routing the continuation through `handle_text` would
have run the composed prompt back through `MAX_PROMPT_CHARS`, which exists to
bound what a person *types*. A long answer is exactly what is worth continuing,
so it would have rejected precisely the cases the button is for. The handlers
now share `_run_text`, which takes an already-validated prompt.

### F4. One GPU budget for two very different workloads

`gpu_seconds_base` was raised to 300 by hand after ZeroGPU killed a run
mid-generation (`release?fail=true` in the Space logs) — correct, and adopted
here. But a single base stretched to cover reasoning charges every quick prompt
for an allocation it will never use, and ZeroGPU quota is spent by the second.

**Fixed:** `gpu_seconds_base_thinking: 300` / `gpu_seconds_base_no_thinking: 90`,
picked by the selected mode. An unknown or missing mode takes the thinking
budget, so a wiring mistake can only over-book, never cut a run short.

### F5. The interface asked for four numbers before it asked for anything else

`max_tokens`, `temperature`, `top_p` and reasoning interact: the first three
mean little without the fourth, and the combination decides whether an answer
arrives at all.

**Fixed:** a `Preset` dropdown sets all four at once, the reasoning choice
stays visible next to it because it is the one that decides whether an answer
fits, and the raw sliders move into a closed `Advanced` accordion. `Custom`
returns Gradio's no-change sentinel rather than values, so selecting it cannot
overwrite dials the user just tuned.

### Not done

**Streaming.** The frozen "40%" progress bar during a 700s generation is a real
complaint, and `TextIteratorStreamer` is the real answer. It also means turning
the handlers into generators and moving a live stream across the ZeroGPU worker
boundary — the same boundary that, three findings ago, could not carry an
exception message. That is its own change with its own failure modes, and
bolting it onto this one would put both at risk.
