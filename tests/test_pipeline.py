"""Tests that run without torch, diffusers or a GPU.

They cover the logic that used to fail silently in production: config
validation, parameter alignment, seed handling, log rotation and the frame
normalisation that decides whether a video file is written correctly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.history import GenerationLogger  # noqa: E402
from pipeline.media import _as_rgb_array, MediaError, prune_directory, save_video  # noqa: E402
from pipeline.models import align, align_frames  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_shipped_config_is_valid():
    config = load_config(REPO_ROOT / "config.json")
    assert config.image["pipeline_class"] == "ChromaPipeline"
    assert config.video["pipeline_class"].startswith("LTX")
    assert set(config.enabled_modules) == {"image", "video", "llm"}


def test_missing_section_is_rejected(tmp_path):
    broken = tmp_path / "config.json"
    broken.write_text(json.dumps({"video": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required section 'image'"):
        load_config(broken)


def test_missing_key_is_reported_by_name(tmp_path):
    raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    del raw["image"]["model_id"]
    broken = tmp_path / "config.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="'image.model_id' is required"):
        load_config(broken)


def test_wrong_type_is_rejected(tmp_path):
    raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    raw["image"]["default_steps"] = "twenty"
    broken = tmp_path / "config.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="must be int"):
        load_config(broken)


def test_inverted_step_range_is_rejected(tmp_path):
    raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    raw["video"]["min_steps"] = 99
    broken = tmp_path / "config.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="exceeds"):
        load_config(broken)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Parameter alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,multiple,minimum,expected",
    [(1024, 64, 512, 1024), (1000, 64, 512, 960), (100, 64, 512, 512), (703, 32, 320, 672)],
)
def test_align(value, multiple, minimum, expected):
    assert align(value, multiple, minimum) == expected


@pytest.mark.parametrize("requested", list(range(1, 200, 7)))
def test_align_frames_always_yields_8n_plus_1(requested):
    frames = align_frames(requested)
    assert frames >= 9
    assert (frames - 1) % 8 == 0


def test_align_frames_never_exceeds_request_beyond_minimum():
    assert align_frames(65) == 65
    assert align_frames(64) == 57
    assert align_frames(2) == 9


# ---------------------------------------------------------------------------
# History log
# ---------------------------------------------------------------------------


def test_log_roundtrip(tmp_path):
    logger = GenerationLogger(str(tmp_path / "log.jsonl"))
    logger.log("image", "a prompt", {"steps": 20}, 1.234, "outputs/x.png", model_id="m")
    entries = logger.history()
    assert len(entries) == 1
    assert entries[0]["type"] == "image"
    assert entries[0]["duration_seconds"] == 1.23
    assert entries[0]["model_id"] == "m"
    assert entries[0]["fallback"] is False


def test_history_is_newest_first_and_bounded(tmp_path):
    logger = GenerationLogger(str(tmp_path / "log.jsonl"))
    for index in range(30):
        logger.log("text", f"prompt {index}", {}, 0.1)
    entries = logger.history(limit=5)
    assert len(entries) == 5
    assert entries[0]["prompt"] == "prompt 29"
    assert entries[-1]["prompt"] == "prompt 25"


def test_corrupt_line_does_not_break_history(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = GenerationLogger(str(path))
    logger.log("image", "good", {}, 0.5)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    entries = logger.history()
    assert [e["prompt"] for e in entries] == ["good"]


def test_log_rotation_caps_file_size(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = GenerationLogger(str(path), max_entries=10)
    for index in range(40):
        logger.log("text", f"p{index}", {}, 0.1)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) <= 12
    assert json.loads(lines[-1])["prompt"] == "p39"


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


def test_rgba_frames_are_reduced_to_rgb():
    frame = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    assert _as_rgb_array(frame).shape == (8, 8, 3)


def test_float_frames_are_rescaled():
    frame = np.zeros((4, 4, 3), dtype=np.float32)
    frame[..., 0] = 1.0
    array = _as_rgb_array(frame)
    assert array.dtype == np.uint8
    assert array[0, 0, 0] == 255


def test_grayscale_frames_are_expanded():
    assert _as_rgb_array(np.zeros((4, 4), dtype=np.uint8)).shape == (4, 4, 3)


def test_save_video_rejects_empty_input(tmp_path):
    with pytest.raises(MediaError, match="no frames"):
        save_video([], str(tmp_path / "out.mp4"))


def test_prune_directory_keeps_newest(tmp_path):
    for index in range(6):
        path = tmp_path / f"f{index}.png"
        path.write_bytes(b"x")
        os.utime(path, (index, index))
    removed = prune_directory(str(tmp_path), keep=2)
    assert removed == 4
    assert {p.name for p in tmp_path.iterdir()} == {"f4.png", "f5.png"}


def test_prune_directory_never_deletes_the_log(tmp_path):
    (tmp_path / "generation_log.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "a.png").write_bytes(b"x")
    prune_directory(str(tmp_path), keep=0)
    assert (tmp_path / "generation_log.jsonl").exists()


# ---------------------------------------------------------------------------
# Manager behaviour that needs no torch
# ---------------------------------------------------------------------------


def _manager(tmp_path, **overrides):
    from pipeline.config import AppConfig
    from pipeline.models import ModelManager

    raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    raw["system"]["output_dir"] = str(tmp_path / "outputs")
    raw["system"]["log_file"] = str(tmp_path / "outputs" / "log.jsonl")
    for section, patch in overrides.items():
        raw[section].update(patch)
    return ModelManager(AppConfig(raw=raw))


def test_disabled_module_reports_clearly(tmp_path):
    from pipeline.models import ModelLoadError

    manager = _manager(tmp_path, video={"enabled": False})
    with pytest.raises(ModelLoadError, match="disabled in config.json"):
        manager.ensure_loaded("video")


def test_unknown_module_is_rejected(tmp_path):
    manager = _manager(tmp_path)
    with pytest.raises(ValueError, match="unknown module"):
        manager.ensure_loaded("audio")


def test_status_before_loading(tmp_path):
    assert _manager(tmp_path).status() == "No model loaded"
    assert _manager(tmp_path).loaded_module is None


def test_unload_without_a_model_is_a_noop(tmp_path):
    _manager(tmp_path).unload()  # must not raise


@pytest.mark.parametrize(
    "message,expected",
    [
        ("401 Client Error: gated repo", "HF_TOKEN"),
        ("404 Client Error: Repository Not Found", "does not exist"),
        ("OSError: No space left on device", "disk"),
        ("CUDA out of memory", "Out of memory"),
        ("something entirely unexpected", "Could not load"),
    ],
)
def test_load_failures_get_actionable_messages(tmp_path, message, expected):
    manager = _manager(tmp_path)
    explanation = manager._explain_load_failure("some/model", RuntimeError(message))
    assert expected in explanation
    assert "some/model" in explanation


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CUDA out of memory", "Lower the resolution"),
        ("GPU task aborted", "allocation expired"),
        ("plain failure", "plain failure"),
    ],
)
def test_generation_failures_get_actionable_messages(tmp_path, message, expected):
    manager = _manager(tmp_path)
    assert expected in manager._explain_generation_failure(RuntimeError(message))


def test_notices_are_drained_once(tmp_path):
    manager = _manager(tmp_path)
    manager._notice("first")
    manager._notice("second")
    assert manager.take_notices() == ["first", "second"]
    assert manager.take_notices() == []


def test_output_paths_are_unique_per_seed(tmp_path):
    manager = _manager(tmp_path)
    first = manager._output_path("image", 1, "png")
    second = manager._output_path("image", 2, "png")
    assert first != second
    assert first.endswith("_1.png") and second.endswith("_2.png")


# ---------------------------------------------------------------------------
# App-level input handling
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_module():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import app as app_module

    return app_module


def test_seed_is_randomised_when_requested(app_module):
    seeds = {app_module._resolve_seed(0, True) for _ in range(20)}
    assert len(seeds) > 1


def test_seed_none_is_tolerated(app_module):
    assert 0 <= app_module._resolve_seed(None, False) <= app_module.SEED_MAX


def test_seed_is_clamped(app_module):
    assert app_module._resolve_seed(-5, False) == 0
    assert app_module._resolve_seed(10**12, False) == app_module.SEED_MAX
    assert app_module._resolve_seed("not a number", False) >= 0


def test_empty_prompt_raises_gradio_error(app_module):
    import gradio as gr

    with pytest.raises(gr.Error):
        app_module._clean_prompt("   ")


def test_overlong_prompt_raises_gradio_error(app_module):
    import gradio as gr

    with pytest.raises(gr.Error):
        app_module._clean_prompt("x" * (app_module.MAX_PROMPT_CHARS + 1))


def test_history_handler_returns_placeholder_row(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module.MANAGER, "logger", GenerationLogger(str(tmp_path / "empty.jsonl"))
    )
    rows = app_module.handle_history()
    assert len(rows) == 1
    assert len(rows[0]) == len(
        __import__("ui_components").HISTORY_HEADERS
    ), "history rows must match the table headers"


def test_history_handler_maps_columns(app_module, tmp_path, monkeypatch):
    logger = GenerationLogger(str(tmp_path / "log.jsonl"))
    logger.log("video", "x" * 300, {}, 12.0, "outputs/v.mp4", model_id="m", fallback=True)
    monkeypatch.setattr(app_module.MANAGER, "logger", logger)
    row = app_module.handle_history()[0]
    assert row[1] == "VIDEO"
    assert row[2] == "m"
    assert len(row[3]) <= 120
    assert row[5] == "v.mp4"
    assert row[6] == "yes"


def test_ui_builds(app_module):
    demo = app_module.build_ui()
    assert demo.blocks
