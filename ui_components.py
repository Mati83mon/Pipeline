"""Gradio UI builders.

Every builder takes the config explicitly. The previous version imported
``config.json`` into a module-level constant at import time, which meant the
"Reload config" button changed the manager's behaviour but left every slider
pinned to the limits captured when the process started.
"""

from __future__ import annotations

from typing import Any, Dict

import gradio as gr

from pipeline.remote import FPS as REMOTE_FPS, estimate_gpu_seconds, snap_frames

HISTORY_HEADERS = [
    "Time (UTC)",
    "Type",
    "Model",
    "Prompt",
    "Seconds",
    "Output",
    "Fallback",
]


# Reasoning choices, in the order a user should meet them. The value on the
# right is the key `ModelManager.THINKING_MODES` understands; the label is what
# it costs you, because that is the decision actually being made.
THINKING_CHOICES = [
    ("Standard — reasoning, no extra prodding", "standard"),
    ("Brief — short reasoning, quicker answer", "brief"),
    ("Deep — validates assumptions, very long", "deep"),
    ("Off — answer immediately, no reasoning", "off"),
]

# Presets exist because `max_tokens` / `temperature` / `top_p` / reasoning are
# four dials whose *interaction* decides whether an answer arrives at all, and
# that is not something to rediscover per prompt. Each is a point that was
# actually tried, not a round number.
TEXT_PRESETS: Dict[str, Dict[str, Any]] = {
    "Quick answer": {
        "max_tokens": 700, "temperature": 0.7, "top_p": 0.9, "thinking": "off",
    },
    "Standard": {
        "max_tokens": 1500, "temperature": 1.0, "top_p": 0.95, "thinking": "standard",
    },
    "Deep dive": {
        "max_tokens": 4096, "temperature": 1.0, "top_p": 0.95, "thinking": "deep",
    },
    "Creative writing": {
        "max_tokens": 2000, "temperature": 1.0, "top_p": 0.95, "thinking": "brief",
    },
}
CUSTOM_PRESET = "Custom"

# The two engines behind the Video tab. The values are what `handle_video`
# dispatches on, so they are part of the contract between this file and app.py.
ENGINE_LTX = "ltx"
ENGINE_H3 = "minimax_h3"


def _seed_row() -> tuple[gr.Number, gr.Checkbox]:
    with gr.Row():
        seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=2**31 - 1)
        randomize = gr.Checkbox(label="Randomise seed", value=True)
    return seed, randomize


def status_row() -> tuple[gr.Textbox, gr.Textbox]:
    with gr.Row():
        model_status = gr.Textbox(
            label="Model", value="No model loaded", interactive=False, scale=3
        )
        runtime_status = gr.Textbox(
            label="Runtime", value="—", interactive=False, scale=2
        )
    return model_status, runtime_status


def disabled_tab(name: str, module: str) -> None:
    with gr.TabItem(name):
        gr.Markdown(
            f"### Module disabled\n"
            f"`{module}` is turned off in `config.json`. Set "
            f"`\"{module}\": {{ \"enabled\": true }}` and restart the Space to use it."
        )


def create_image_tab(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Labelled "Photo"; the config section, the manager's module key and this
    # builder's name stay "image", which is what every other file calls it.
    with gr.TabItem("Photo"):
        gr.Markdown(
            f"**{cfg['model_id']}** · text-to-image. "
            f"Width and height are rounded down to multiples of "
            f"{cfg.get('size_step', 64)}."
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the image…")
                negative = gr.Textbox(
                    label="Negative prompt",
                    lines=2,
                    value=cfg.get("default_negative_prompt", ""),
                )
                with gr.Accordion("Parameters", open=True):
                    steps = gr.Slider(
                        cfg["min_steps"], cfg["max_steps"], cfg["default_steps"],
                        step=1, label="Steps",
                    )
                    guidance = gr.Slider(
                        cfg["min_guidance"], cfg["max_guidance"], cfg["default_guidance"],
                        step=0.1, label="Guidance scale",
                    )
                    width = gr.Slider(
                        cfg.get("min_size", 512), cfg.get("max_size", 1536),
                        cfg["default_width"], step=cfg.get("size_step", 64), label="Width",
                    )
                    height = gr.Slider(
                        cfg.get("min_size", 512), cfg.get("max_size", 1536),
                        cfg["default_height"], step=cfg.get("size_step", 64), label="Height",
                    )
                    seed, randomize = _seed_row()
                button = gr.Button("Generate image", variant="primary")
            with gr.Column(scale=1):
                output = gr.Image(label="Result", type="pil", format="png", height=520)
                info = gr.Textbox(label="Run info", interactive=False, lines=2)
    return {
        "prompt": prompt, "negative": negative, "steps": steps, "guidance": guidance,
        "width": width, "height": height, "seed": seed, "randomize": randomize,
        "button": button, "output": output, "info": info,
    }


def describe_remote_request(canvas: str, duration: float, steps: int, keyframes: int) -> str:
    """What this MiniMax-H3 request will actually cost, before the click.

    Worth its own readout because two of the three numbers are surprising. The
    duration is not what the slider says — the video VAE decodes ``17n + 5``
    frames at 24 fps, so the request is snapped up. And the GPU seconds grow
    quadratically with the canvas, because the model attends over the whole
    packed sequence every layer: 5 s at the fast canvas is about a minute,
    14 s at the full one is several hundred seconds, and the remote Space books
    ``xlarge``, where every second costs two of quota.
    """
    frames = snap_frames(duration)
    seconds = estimate_gpu_seconds(canvas, duration, steps, keyframes)
    return (
        f"{frames} frames · {frames / REMOTE_FPS:.2f} s of video · "
        f"~{seconds} s of GPU booked (xlarge, so ~{seconds * 2} s of quota)"
    )


def _remote_video_group(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The MiniMax-H3 controls. No negative prompt: it is guidance-distilled."""
    from pipeline.remote import RemoteVideoEngine

    engine = RemoteVideoEngine(cfg)
    canvases = engine.canvases()
    default_canvas = engine.default_canvas()
    low, high = engine.duration_bounds()
    loras = engine.loras()
    default_lora = cfg.get("default_lora") if cfg.get("default_lora") in loras else next(iter(loras))
    default_steps = loras[default_lora]
    default_duration = float(cfg.get("default_duration", min(high, 5)))

    gr.Markdown(
        f"**{cfg.get('space_id', '—')}** · runs on a separate Space, so none of "
        f"its 77.3 GB touches this one. Audio is generated with the video in one "
        f"pass. No negative prompt — the model is guidance-distilled. "
        f"**{high:g} s is the hard ceiling**, set by the checkpoint."
    )
    with gr.Row():
        first_image = gr.Image(
            label="First keyframe (optional)", type="filepath",
            sources=["upload", "clipboard"], height=200,
        )
        last_image = gr.Image(
            label="Last keyframe (optional)", type="filepath",
            sources=["upload", "clipboard"], height=200,
        )
    canvas = gr.Dropdown(choices=canvases, value=default_canvas, label="Canvas")
    duration = gr.Slider(
        low, high, default_duration, step=0.5,
        label="Duration (seconds)",
        info=f"Snapped up to the next decodable frame count. {high:g} s is the maximum.",
    )
    lora = gr.Dropdown(
        choices=[(f"{name} ({steps} steps)", name) for name, steps in loras.items()],
        value=default_lora, label="Turbo LoRA",
    )
    steps = gr.Slider(
        cfg.get("min_steps", 2), cfg.get("max_steps", 40), default_steps,
        step=1, label="Sampling steps",
    )
    upsample = gr.Checkbox(
        label="Let the remote conditioner rewrite the prompt",
        value=bool(cfg.get("default_upsample", False)),
    )
    cost = gr.Textbox(
        label="This request",
        value=describe_remote_request(default_canvas, default_duration, default_steps, 0),
        interactive=False, lines=2,
    )

    # Picking a LoRA moves the step count to the one it was distilled for; the
    # slider stays free afterwards, because `off` at 6 steps is a legitimate
    # (if ugly) thing to try.
    lora.change(fn=lambda name: loras.get(name, gr.update()), inputs=[lora], outputs=[steps])

    priced = [canvas, duration, steps, first_image, last_image]
    for control in priced:
        control.change(
            fn=lambda c, d, s, a, b: describe_remote_request(
                c, d, s, int(bool(a)) + int(bool(b))
            ),
            inputs=priced,
            outputs=[cost],
        )

    return {
        "first_image": first_image, "last_image": last_image, "canvas": canvas,
        "duration": duration, "steps": steps, "lora": lora, "upsample": upsample,
        "cost": cost,
    }


def create_video_tab(cfg: Dict[str, Any]) -> Dict[str, Any]:
    remote_cfg = cfg.get("remote_engine") or {}
    remote_on = bool(remote_cfg.get("enabled")) and bool(remote_cfg.get("space_id"))

    with gr.TabItem("Video"):
        if remote_on:
            gr.Markdown(
                "Two engines. **LTX-Video** is local — fast, silent, every "
                "resolution. **MiniMax-H3** runs on another Space and returns "
                "video with a matching soundtrack, up to "
                f"{remote_cfg.get('max_duration', 14):g} seconds."
            )
        else:
            gr.Markdown(
                f"**{cfg['model_id']}** · text-to-video and image-to-video. "
                f"Resolution snaps to multiples of {cfg.get('size_step', 32)}, "
                f"frame count to 8n+1. Attach an image to drive image-to-video."
            )
        with gr.Row():
            with gr.Column(scale=1):
                engine = (
                    gr.Radio(
                        choices=[
                            ("LTX-Video · local, no audio", ENGINE_LTX),
                            (remote_cfg.get("label") or "MiniMax-H3 · remote", ENGINE_H3),
                        ],
                        value=ENGINE_LTX,
                        label="Engine",
                    )
                    if remote_on
                    else gr.State(ENGINE_LTX)
                )
                prompt = gr.Textbox(
                    label="Prompt", lines=4,
                    placeholder="Describe the scene and its motion…",
                )

                with gr.Group(visible=True) as ltx_group:
                    negative = gr.Textbox(
                        label="Negative prompt", lines=2,
                        value=cfg.get("default_negative_prompt", ""),
                    )
                    image = gr.Image(
                        label="Conditioning image (optional)", type="pil",
                        sources=["upload", "clipboard"], height=220,
                    )
                    with gr.Accordion("Parameters", open=True):
                        width = gr.Slider(
                            cfg.get("min_size", 320), cfg.get("max_size", 1216),
                            cfg["default_width"], step=cfg.get("size_step", 32),
                            label="Width",
                        )
                        height = gr.Slider(
                            cfg.get("min_size", 320), cfg.get("max_size", 1216),
                            cfg["default_height"], step=cfg.get("size_step", 32),
                            label="Height",
                        )
                        frames = gr.Slider(
                            cfg["min_frames"], cfg["max_frames"], cfg["default_frames"],
                            step=8, label="Frames",
                        )
                        fps = gr.Slider(
                            cfg["min_fps"], cfg["max_fps"], cfg["default_fps"],
                            step=1, label="Playback FPS",
                        )
                        steps = gr.Slider(
                            cfg["min_steps"], cfg["max_steps"], cfg["default_steps"],
                            step=1, label="Steps",
                        )
                        guidance = gr.Slider(
                            cfg.get("min_guidance", 1.0), cfg.get("max_guidance", 10.0),
                            cfg.get("default_guidance", 3.0), step=0.1,
                            label="Guidance scale",
                        )

                remote: Dict[str, Any] = {}
                with gr.Group(visible=False) as h3_group:
                    if remote_on:
                        remote = _remote_video_group(remote_cfg)

                seed, randomize = _seed_row()
                ltx_button = gr.Button("Generate video", variant="primary")
                h3_button = gr.Button(
                    "Generate video + audio", variant="primary", visible=False
                )
            with gr.Column(scale=1):
                output = gr.Video(label="Result", height=520)
                info = gr.Textbox(label="Run info", interactive=False, lines=3)

        if remote_on:
            # One button per engine rather than one that branches: the two
            # events need different concurrency. A remote render holds no local
            # GPU and takes minutes, so serialising it behind the local queue
            # would freeze every other tab for the duration.
            engine.change(
                fn=toggle_video_engine,
                inputs=[engine],
                outputs=[ltx_group, h3_group, ltx_button, h3_button],
            )

    built = {
        "engine": engine, "prompt": prompt, "negative": negative, "image": image,
        "width": width, "height": height, "frames": frames, "fps": fps,
        "steps": steps, "guidance": guidance, "seed": seed, "randomize": randomize,
        "button": ltx_button, "h3_button": h3_button, "output": output, "info": info,
        "ltx_group": ltx_group, "h3_group": h3_group, "remote_enabled": remote_on,
    }
    built.update({f"h3_{key}": value for key, value in remote.items()})
    return built


def toggle_video_engine(engine: str) -> tuple[Any, Any, Any, Any]:
    """Show the controls and the button belonging to the selected engine."""
    local = engine != ENGINE_H3
    return (
        gr.update(visible=local),
        gr.update(visible=not local),
        gr.update(visible=local),
        gr.update(visible=not local),
    )


def create_text_tab(cfg: Dict[str, Any]) -> Dict[str, Any]:
    with gr.TabItem("Text"):
        fallback = cfg.get("fallback_model_id")
        gr.Markdown(
            f"**{cfg['model_id']}** · vision-language chat."
            + (f" Falls back to **{fallback}** when the primary model is "
               f"unavailable (gated, out of memory, unsupported)." if fallback else "")
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Ask anything, or describe what to do with the attached media…")
                with gr.Row():
                    image = gr.Image(
                        label="Image (optional)", type="pil",
                        sources=["upload", "clipboard"], height=200,
                    )
                    video = gr.Video(label="Video (optional)", sources=["upload"], height=200)
                with gr.Row():
                    preset = gr.Dropdown(
                        choices=[*TEXT_PRESETS, CUSTOM_PRESET],
                        value="Standard",
                        label="Preset",
                    )
                    thinking = gr.Dropdown(
                        choices=THINKING_CHOICES,
                        value=cfg.get("default_thinking", "standard"),
                        label="Reasoning",
                        info=(
                            "This model reasons before answering. On long prompts "
                            "that can consume the whole token budget before the "
                            "answer starts."
                        ),
                    )
                with gr.Accordion("Advanced", open=False):
                    max_tokens = gr.Slider(
                        cfg["min_tokens"], cfg["max_tokens"], cfg["default_max_tokens"],
                        step=32, label="Max new tokens",
                    )
                    temperature = gr.Slider(
                        cfg.get("min_temperature", 0.0), cfg.get("max_temperature", 2.0),
                        cfg["default_temperature"], step=0.05,
                        label="Temperature (0 = greedy)",
                    )
                    top_p = gr.Slider(0.05, 1.0, cfg["default_top_p"], step=0.05, label="Top-p")
                    seed, randomize = _seed_row()
                with gr.Row():
                    button = gr.Button("Generate text", variant="primary")
                    continue_button = gr.Button("Continue last answer")
            with gr.Column(scale=1):
                output = gr.Textbox(label="Response", lines=22)
                info = gr.Textbox(label="Run info", interactive=False, lines=3)

    # Picking a preset moves the dials it owns and leaves everything else alone.
    preset.change(
        fn=apply_text_preset,
        inputs=[preset],
        outputs=[max_tokens, temperature, top_p, thinking],
    )
    return {
        "prompt": prompt, "image": image, "video": video, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "seed": seed,
        "randomize": randomize, "button": button, "output": output, "info": info,
        "thinking": thinking, "preset": preset, "continue_button": continue_button,
    }


def apply_text_preset(name: str) -> tuple[Any, Any, Any, Any]:
    """Map a preset name onto the four controls it sets.

    ``Custom`` means "leave what I set", so it returns `gr.update()` sentinels
    rather than values — otherwise selecting it would silently reset the dials
    the user had just tuned by hand.
    """
    values = TEXT_PRESETS.get(name)
    if values is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    return (
        values["max_tokens"],
        values["temperature"],
        values["top_p"],
        values["thinking"],
    )


def create_history_tab(limit: int) -> Dict[str, Any]:
    with gr.TabItem("History"):
        gr.Markdown(
            f"Last {limit} runs, newest first. Stored in the Space's ephemeral "
            f"disk, so it resets when the Space restarts."
        )
        refresh = gr.Button("Refresh")
        table = gr.Dataframe(
            headers=HISTORY_HEADERS,
            datatype=["str", "str", "str", "str", "number", "str", "str"],
            interactive=False,
            wrap=True,
        )
    return {"refresh": refresh, "table": table}


def create_system_panel() -> Dict[str, Any]:
    with gr.Accordion("System", open=False):
        gr.Markdown(
            "Models load on first use and only one stays resident. The first "
            "run of each tab downloads tens of gigabytes and can take several "
            "minutes."
        )
        with gr.Row():
            unload = gr.Button("Unload model")
            refresh_status = gr.Button("Refresh status")
        log = gr.Textbox(label="System log", interactive=False, lines=4)
    return {"unload": unload, "refresh_status": refresh_status, "log": log}
