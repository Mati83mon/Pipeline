"""AI Media Pipeline — Gradio entry point.

Image (Chroma1-HD) · Video (LTX-Video) · Vision-language chat (Qwen3.8-27B).

Runs unchanged on ZeroGPU, on a dedicated GPU Space, in Docker and on CPU.
See docs/AUDIT.md for the defects this rewrite fixes.
"""

from __future__ import annotations

# `spaces` must be imported before torch. pipeline.runtime does that and does
# not pull torch in at import time, so importing it first is enough.
from pipeline.runtime import (  # isort: skip
    IS_ZEROGPU,
    describe_runtime,
    free_vram,
    gpu_task,
    hf_token,
    vram_status,
)

import functools
import logging
import os
import random
from typing import Any, List, Optional

import gradio as gr

import ui_components as ui
from pipeline.config import ConfigError, load_config
from pipeline.models import GenerationError, ModelLoadError, ModelManager

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("app")

CONFIG_PATH = os.environ.get("PIPELINE_CONFIG", "config.json")
CONFIG = load_config(CONFIG_PATH)
MANAGER = ModelManager(CONFIG)

MAX_PROMPT_CHARS = int(CONFIG.system.get("max_prompt_chars", 4000))
SEED_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clean_prompt(prompt: Optional[str], field: str = "Prompt") -> str:
    text = (prompt or "").strip()
    if not text:
        raise gr.Error(f"{field} cannot be empty.")
    if len(text) > MAX_PROMPT_CHARS:
        raise gr.Error(f"{field} is longer than {MAX_PROMPT_CHARS} characters.")
    return text


def _resolve_seed(seed: Any, randomize: bool) -> int:
    """Gradio hands back ``None`` for a cleared Number field."""
    if randomize or seed is None:
        return random.randint(0, SEED_MAX)
    try:
        return max(0, min(SEED_MAX, int(seed)))
    except (TypeError, ValueError):
        return random.randint(0, SEED_MAX)


def _status() -> tuple[str, str]:
    return MANAGER.status(), f"{describe_runtime()} · {vram_status()}"


def _run_info(kind: str, seed: int, duration: float, path: str) -> str:
    notices = MANAGER.take_notices()
    line = (
        f"{kind} · seed {seed} · {duration:.1f}s · "
        f"{MANAGER.status()} · saved {os.path.basename(path)}"
    )
    return line + ("\n" + "\n".join(notices) if notices else "")


def _fail(exc: Exception) -> "gr.Error":
    """Turn a backend exception into a user-visible Gradio error.

    `ModelLoadError` and `GenerationError` are already logged with a full
    traceback by the manager that raised them, so re-logging here would only
    duplicate it. Anything else has not been logged at all yet.
    """
    if isinstance(exc, gr.Error):
        return exc
    if isinstance(exc, (ModelLoadError, GenerationError)):
        return gr.Error(str(exc))
    LOGGER.error("request failed", exc_info=exc)
    return gr.Error(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# GPU tasks — no Gradio *components* cross this boundary, but `gr.Error` must
# ---------------------------------------------------------------------------


def _gpu_safe(fn):
    """Convert a failure into `gr.Error` *inside* the ZeroGPU worker.

    ZeroGPU forks a worker process and ships the result back over a queue.
    `spaces.zero.wrappers.exception_result` keeps the exception object itself
    only when it is a `gr.Error`::

        if isinstance(exc, gr.Error):
            gradio_error = exc
        return ExceptionResult(traceback=..., error_cls=exc.__class__.__name__,
                               gradio_error=gradio_error)

    Everything else arrives in the parent as nothing but a class name, which
    it re-raises as `error("ZeroGPU worker error", res.error_cls)`. That is how
    a `GenerationError` carrying `ImportError: finegrained-fp8 kernel
    unavailable: ...` reached the UI as the useless `Error: 'GenerationError'`.

    The message only exists inside the worker, so that is the only place it can
    be attached to something that survives the trip. Off ZeroGPU there is no
    boundary and the wrapper is simply a no-op re-raise.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise _fail(exc) from exc

    return wrapper


def _image_duration(prompt, negative, steps, guidance, width, height, seed) -> float:
    cfg = CONFIG.image
    return float(cfg.get("gpu_seconds_base", 45)) + float(
        cfg.get("gpu_seconds_per_step", 1.6)
    ) * float(steps or 1)


@gpu_task(duration=_image_duration)
@_gpu_safe
def _gpu_image(prompt, negative, steps, guidance, width, height, seed):
    return MANAGER.generate_image(
        prompt=prompt,
        negative_prompt=negative,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed,
    )


def _video_duration(
    prompt, negative, image, width, height, frames, fps, steps, guidance, seed
) -> float:
    cfg = CONFIG.video
    return (
        float(cfg.get("gpu_seconds_base", 60))
        + float(cfg.get("gpu_seconds_per_step", 2.0)) * float(steps or 1)
        + float(cfg.get("gpu_seconds_per_frame", 0.35)) * float(frames or 1)
    )


@gpu_task(duration=_video_duration)
@_gpu_safe
def _gpu_video(prompt, negative, image, width, height, frames, fps, steps, guidance, seed):
    return MANAGER.generate_video(
        prompt=prompt,
        negative_prompt=negative,
        image=image,
        num_frames=frames,
        fps=fps,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed,
    )


def _text_duration(
    prompt, image, video, max_tokens, temperature, top_p, seed, thinking=None
) -> float:
    """Reasoning costs wall clock that the token budget alone does not predict.

    A single base stretched to cover the worst case buys every quick prompt a
    ZeroGPU allocation it will never use, and quota is charged by the second.
    With reasoning off the model starts emitting the answer immediately, so it
    gets the smaller base.
    """
    cfg = CONFIG.llm
    fallback = cfg.get("gpu_seconds_base", 300)
    key = (
        "gpu_seconds_base_no_thinking"
        if str(thinking or "").lower() == "off"
        else "gpu_seconds_base_thinking"
    )
    return float(cfg.get(key, fallback)) + float(
        cfg.get("gpu_seconds_per_100_tokens", 10.0)
    ) * (float(max_tokens or 100) / 100.0)


@gpu_task(duration=_text_duration)
@_gpu_safe
def _gpu_text(prompt, image, video, max_tokens, temperature, top_p, seed, thinking=None):
    return MANAGER.generate_text(
        prompt=prompt,
        image=image,
        video=video,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        thinking=thinking,
    )


# ---------------------------------------------------------------------------
# Gradio handlers — model loading happens here, outside the GPU allocation
# ---------------------------------------------------------------------------


def handle_image(
    prompt, negative, steps, guidance, width, height, seed, randomize,
    progress=gr.Progress(),
):
    prompt = _clean_prompt(prompt)
    seed = _resolve_seed(seed, randomize)
    try:
        progress(0.1, desc="Preparing model (first run downloads ~28 GB)…")
        MANAGER.ensure_loaded("image")
        eta = int(_image_duration(prompt, negative, steps, guidance, width, height, seed))
        progress(0.4, desc=f"Generating on GPU (up to ~{eta}s)…")
        image, path, duration = _gpu_image(
            prompt, negative, int(steps), float(guidance), int(width), int(height), seed
        )
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from exc
    progress(1.0, desc="Done")
    return image, _run_info("image", seed, duration, path), *_status()


def handle_video(
    prompt, negative, image, width, height, frames, fps, steps, guidance, seed, randomize,
    progress=gr.Progress(),
):
    prompt = _clean_prompt(prompt)
    seed = _resolve_seed(seed, randomize)
    try:
        progress(0.1, desc="Preparing model (first run downloads ~27 GB)…")
        MANAGER.ensure_loaded("video")
        eta = int(
            _video_duration(
                prompt, negative, image, width, height, frames, fps, steps, guidance, seed
            )
        )
        progress(0.4, desc=f"Rendering on GPU (up to ~{eta}s)…")
        path, duration = _gpu_video(
            prompt, negative, image, int(width), int(height), int(frames),
            int(fps), int(steps), float(guidance), seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from exc
    progress(1.0, desc="Done")
    return path, _run_info("video", seed, duration, path), *_status()


CONTINUATION_TEMPLATE = (
    "Here is the beginning of your previous answer, which was cut off before "
    "it finished:\n\n---\n{previous}\n---\n\nContinue from exactly where it "
    "stops, in the same style and language. Do not repeat anything already "
    "written and do not start over.\n\nThe original request was:\n{prompt}"
)


def _run_text(
    prompt, image, video, max_tokens, temperature, top_p, seed, randomize, thinking,
    progress,
):
    """Shared body of the two text handlers.

    Takes the prompt already validated. `MAX_PROMPT_CHARS` guards what a person
    types; a continuation prompt is assembled by this app from an answer the
    model itself just produced, and re-checking it against a limit meant for
    typed input would reject exactly the long answers worth continuing.
    """
    seed = _resolve_seed(seed, randomize)
    try:
        progress(0.1, desc="Preparing model (first run downloads ~31 GB)…")
        MANAGER.ensure_loaded("llm")
        eta = int(
            _text_duration(
                prompt, image, video, max_tokens, temperature, top_p, seed, thinking
            )
        )
        progress(0.4, desc=f"Generating on GPU (up to ~{eta}s)…")
        response, path, duration = _gpu_text(
            prompt, image, video, int(max_tokens), float(temperature), float(top_p),
            seed, thinking,
        )
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from exc
    progress(1.0, desc="Done")
    return response, _run_info("text", seed, duration, path), *_status(), response


def handle_text(
    prompt, image, video, max_tokens, temperature, top_p, seed, randomize, thinking,
    progress=gr.Progress(),
):
    return _run_text(
        _clean_prompt(prompt), image, video, max_tokens, temperature, top_p, seed,
        randomize, thinking, progress,
    )


def handle_continue(
    prompt, previous, image, video, max_tokens, temperature, top_p, seed, randomize,
    thinking, progress=gr.Progress(),
):
    """Resume a cut-off answer by feeding it back in.

    The Text tab is stateless by design — one model resident at a time, no
    session store — so "carry on" means nothing to the model unless the text to
    carry on from travels with the prompt. This is the whole of that state: the
    last answer, held in a `gr.State` for exactly this button.
    """
    if not (previous or "").strip():
        raise gr.Error("Nothing to continue yet — generate an answer first.")
    return _run_text(
        CONTINUATION_TEMPLATE.format(
            previous=previous.strip(), prompt=_clean_prompt(prompt)
        ),
        image, video, max_tokens, temperature, top_p, seed, randomize, thinking,
        progress,
    )


def handle_history() -> List[List[Any]]:
    limit = int(CONFIG.system.get("history_limit", 25))
    rows: List[List[Any]] = []
    for entry in MANAGER.logger.history(limit=limit):
        prompt = entry.get("prompt", "") or ""
        rows.append(
            [
                (entry.get("timestamp", "") or "")[11:19] or "—",
                str(entry.get("type", "")).upper(),
                entry.get("model_id") or "—",
                prompt[:117] + "…" if len(prompt) > 120 else prompt,
                entry.get("duration_seconds", 0),
                os.path.basename(entry.get("output_path") or "") or "—",
                "yes" if entry.get("fallback") else "no",
            ]
        )
    return rows or [["—", "—", "—", "No runs yet", 0, "—", "—"]]


def handle_unload() -> tuple[str, str, str]:
    MANAGER.unload()
    free_vram()
    return "Model unloaded and VRAM released.", *_status()


def handle_refresh_status() -> tuple[str, str, str]:
    notices = MANAGER.take_notices()
    return ("\n".join(notices) or "Status refreshed."), *_status()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


THEME = gr.themes.Soft(primary_hue="violet", neutral_hue="slate")


def build_ui() -> gr.Blocks:
    banner = (
        "Running on **ZeroGPU** — a GPU is attached per request, so the first "
        "run of each tab also downloads the model."
        if IS_ZEROGPU
        else f"Runtime: {describe_runtime()}"
    )
    token_note = (
        ""
        if hf_token()
        else "\n\n> No `HF_TOKEN` secret is set. Gated models (including the "
        "default text model) will fall back to their open alternative."
    )

    with gr.Blocks(title="AI Media Pipeline") as demo:
        gr.Markdown(
            "# AI Media Pipeline\n"
            "Image · Video · Vision-language text, one model resident at a time.\n\n"
            f"{banner}{token_note}"
        )
        model_status, runtime_status = ui.status_row()
        status_targets = [model_status, runtime_status]

        with gr.Tabs():
            image_ui = (
                ui.create_image_tab(CONFIG.image)
                if CONFIG.is_enabled("image")
                else ui.disabled_tab("Image", "image")
            )
            video_ui = (
                ui.create_video_tab(CONFIG.video)
                if CONFIG.is_enabled("video")
                else ui.disabled_tab("Video", "video")
            )
            text_ui = (
                ui.create_text_tab(CONFIG.llm)
                if CONFIG.is_enabled("llm")
                else ui.disabled_tab("Text", "llm")
            )
            history_ui = ui.create_history_tab(int(CONFIG.system.get("history_limit", 25)))

        system_ui = ui.create_system_panel()

        if image_ui:
            image_ui["button"].click(
                fn=handle_image,
                inputs=[
                    image_ui["prompt"], image_ui["negative"], image_ui["steps"],
                    image_ui["guidance"], image_ui["width"], image_ui["height"],
                    image_ui["seed"], image_ui["randomize"],
                ],
                outputs=[image_ui["output"], image_ui["info"], *status_targets],
            )
        if video_ui:
            video_ui["button"].click(
                fn=handle_video,
                inputs=[
                    video_ui["prompt"], video_ui["negative"], video_ui["image"],
                    video_ui["width"], video_ui["height"], video_ui["frames"],
                    video_ui["fps"], video_ui["steps"], video_ui["guidance"],
                    video_ui["seed"], video_ui["randomize"],
                ],
                outputs=[video_ui["output"], video_ui["info"], *status_targets],
            )
        if text_ui:
            # The only state the Text tab keeps: the last answer, so
            # "Continue last answer" has something to continue.
            last_answer = gr.State("")
            text_inputs = [
                text_ui["prompt"], text_ui["image"], text_ui["video"],
                text_ui["max_tokens"], text_ui["temperature"], text_ui["top_p"],
                text_ui["seed"], text_ui["randomize"], text_ui["thinking"],
            ]
            text_outputs = [
                text_ui["output"], text_ui["info"], *status_targets, last_answer,
            ]
            text_ui["button"].click(
                fn=handle_text, inputs=text_inputs, outputs=text_outputs
            )
            text_ui["continue_button"].click(
                fn=handle_continue,
                inputs=[text_inputs[0], last_answer, *text_inputs[1:]],
                outputs=text_outputs,
            )

        history_ui["refresh"].click(fn=handle_history, outputs=[history_ui["table"]])
        system_ui["unload"].click(
            fn=handle_unload, outputs=[system_ui["log"], *status_targets]
        )
        system_ui["refresh_status"].click(
            fn=handle_refresh_status, outputs=[system_ui["log"], *status_targets]
        )

        demo.load(fn=handle_history, outputs=[history_ui["table"]])
        demo.load(fn=_status, outputs=status_targets)

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=int(CONFIG.system.get("queue_max_size", 8)))
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860))),
        show_error=True,
        theme=THEME,
    )


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
