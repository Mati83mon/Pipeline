"""Gradio UI builders.

Every builder takes the config explicitly. The previous version imported
``config.json`` into a module-level constant at import time, which meant the
"Reload config" button changed the manager's behaviour but left every slider
pinned to the limits captured when the process started.
"""

from __future__ import annotations

from typing import Any, Dict

import gradio as gr

HISTORY_HEADERS = [
    "Time (UTC)",
    "Type",
    "Model",
    "Prompt",
    "Seconds",
    "Output",
    "Fallback",
]


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
    with gr.TabItem("Image"):
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


def create_video_tab(cfg: Dict[str, Any]) -> Dict[str, Any]:
    with gr.TabItem("Video"):
        gr.Markdown(
            f"**{cfg['model_id']}** · text-to-video and image-to-video. "
            f"Resolution snaps to multiples of {cfg.get('size_step', 32)}, "
            f"frame count to 8n+1. Attach an image to drive image-to-video."
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the scene and its motion…")
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
                        cfg["default_width"], step=cfg.get("size_step", 32), label="Width",
                    )
                    height = gr.Slider(
                        cfg.get("min_size", 320), cfg.get("max_size", 1216),
                        cfg["default_height"], step=cfg.get("size_step", 32), label="Height",
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
                        cfg.get("default_guidance", 3.0), step=0.1, label="Guidance scale",
                    )
                    seed, randomize = _seed_row()
                button = gr.Button("Generate video", variant="primary")
            with gr.Column(scale=1):
                output = gr.Video(label="Result", height=520)
                info = gr.Textbox(label="Run info", interactive=False, lines=2)
    return {
        "prompt": prompt, "negative": negative, "image": image, "width": width,
        "height": height, "frames": frames, "fps": fps, "steps": steps,
        "guidance": guidance, "seed": seed, "randomize": randomize,
        "button": button, "output": output, "info": info,
    }


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
                with gr.Accordion("Parameters", open=True):
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
                button = gr.Button("Generate text", variant="primary")
            with gr.Column(scale=1):
                output = gr.Textbox(label="Response", lines=22)
                info = gr.Textbox(label="Run info", interactive=False, lines=2)
    return {
        "prompt": prompt, "image": image, "video": video, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "seed": seed,
        "randomize": randomize, "button": button, "output": output, "info": info,
    }


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
