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
# Space metadata
#
# The original draft shipped a README with no YAML front matter at all, so the
# Space had no `sdk` / `app_file` and could never build (docs/AUDIT.md, A1).
# These tests make that class of mistake — and a drifting sdk_version — a test
# failure rather than a deployment failure.
# ---------------------------------------------------------------------------


def _front_matter(text: str) -> dict:
    """Parse the top-level keys of a Space README's YAML front matter.

    Deliberately dependency-free: it only needs `key: value` and simple
    `- item` lists, which is all a Space card uses.
    """
    if not text.startswith("---"):
        raise AssertionError("README.md must open with a YAML front matter block")
    _, _, rest = text.partition("---\n")
    block, sep, _ = rest.partition("\n---")
    if not sep:
        raise AssertionError("README.md front matter is not terminated by '---'")

    parsed: dict = {}
    current_list_key = None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_list_key:
            parsed[current_list_key].append(stripped[2:].strip())
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value:
            parsed[key] = value
            current_list_key = None
        else:
            parsed[key] = []
            current_list_key = key
    return parsed


@pytest.fixture(scope="module")
def front_matter() -> dict:
    return _front_matter((REPO_ROOT / "README.md").read_text(encoding="utf-8"))


def test_space_front_matter_has_required_keys(front_matter):
    for key in ("title", "sdk", "sdk_version", "app_file", "python_version"):
        assert key in front_matter, f"README front matter is missing '{key}'"
    assert front_matter["sdk"] == "gradio"


def test_app_file_exists(front_matter):
    assert (REPO_ROOT / front_matter["app_file"]).is_file()


def test_python_version_is_supported_by_zerogpu(front_matter):
    # ZeroGPU only offers these two interpreters.
    assert front_matter["python_version"] in {"3.10.13", "3.12.12"}


def _pinned_version(package: str) -> str:
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line.startswith(f"{package}=="):
            return line.split("==", 1)[1].strip()
    raise AssertionError(f"{package} is not pinned in requirements.txt")


def test_sdk_version_matches_pinned_gradio(front_matter):
    assert front_matter["sdk_version"] == _pinned_version("gradio"), (
        "README sdk_version and the gradio pin in requirements.txt must agree, "
        "or the Space builds one Gradio and installs another"
    )


def test_torch_pin_is_supported_by_zerogpu():
    assert _pinned_version("torch") in {"2.8.0", "2.9.1", "2.10.0", "2.11.0"}


def test_declared_models_cover_the_configured_ones(front_matter):
    declared = set(front_matter.get("models", []))
    config = load_config(REPO_ROOT / "config.json")
    for section in ("image", "video", "llm"):
        block = config.section(section)
        assert block["model_id"] in declared, (
            f"{block['model_id']} is used by the app but not listed in the "
            f"README front matter"
        )
        fallback = block.get("fallback_model_id")
        if fallback:
            assert fallback in declared


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


def _fake_transformers(quantization_config, record):
    """A stand-in transformers module that records how the model was loaded."""
    import types

    module = types.ModuleType("transformers")

    class _Config:
        pass

    config = _Config()
    if quantization_config is not None:
        config.quantization_config = quantization_config

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return config

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return object()

    class _Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            record.append(kwargs)

            class _Model:
                device = "cpu"

                def eval(self):
                    return self

            return _Model()

    module.AutoConfig = AutoConfig
    module.AutoProcessor = AutoProcessor
    module.AutoModelForImageTextToText = _Loader
    module.AutoModelForCausalLM = _Loader
    module.AutoTokenizer = _Loader
    return module


def test_declared_quantization_is_detected(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers({"quant_method": "fp8", "weight_block_size": [128, 128]}, []),
    )
    assert manager._declared_quantization("some/model", False, None) == "fp8"


def test_no_declared_quantization_returns_none(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(None, []))
    assert manager._declared_quantization("some/model", False, None) is None


def test_unreadable_config_does_not_raise(tmp_path, monkeypatch):
    import types

    manager = _manager(tmp_path)
    module = types.ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            raise OSError("401 gated")

    module.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", module)
    assert manager._declared_quantization("some/model", False, None) is None


def test_prequantised_checkpoint_is_loaded_without_a_forced_dtype(tmp_path, monkeypatch):
    """The regression this guards: forcing bfloat16 onto block-FP8 weights."""
    record: list = []
    manager = _manager(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers({"quant_method": "fp8"}, record),
    )
    manager._load_llm("some/model", {"dtype": "bfloat16", "multimodal": True})
    assert record, "the loader was never called"
    assert "dtype" not in record[-1]
    assert "torch_dtype" not in record[-1]


def _fake_torch():
    """Enough of torch for resolve_dtype, so this path needs no real install."""
    import types

    module = types.ModuleType("torch")
    module.bfloat16 = "bfloat16"
    module.float16 = "float16"
    module.float32 = "float32"

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def is_bf16_supported():
            return False

    module.cuda = _Cuda
    return module


def test_plain_checkpoint_still_gets_an_explicit_dtype(tmp_path, monkeypatch):
    record: list = []
    manager = _manager(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(None, record))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    manager._load_llm("some/model", {"dtype": "bfloat16", "multimodal": True})
    assert record, "the loader was never called"
    assert "dtype" in record[-1] or "torch_dtype" in record[-1]


def test_bitsandbytes_is_not_stacked_on_a_quantised_checkpoint(tmp_path, monkeypatch):
    record: list = []
    manager = _manager(tmp_path)
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers({"quant_method": "fp8"}, record)
    )
    manager._load_llm(
        "some/model", {"dtype": "bfloat16", "multimodal": True, "load_in_4bit": True}
    )
    assert "quantization_config" not in record[-1]
    assert any("already fp8-quantised" in n for n in manager.take_notices())


@pytest.mark.parametrize(
    "exc,expected",
    [
        (RuntimeError("boom"), "RuntimeError: boom"),
        (ValueError(""), "ValueError"),
        (OSError("a\nb  c"), "OSError: a b c"),
    ],
)
def test_short_reason_is_one_informative_line(tmp_path, exc, expected):
    assert _manager(tmp_path)._short_reason(exc) == expected


def test_short_reason_truncates(tmp_path):
    reason = _manager(tmp_path)._short_reason(RuntimeError("x" * 500))
    assert len(reason) < 300
    assert reason.endswith("…")


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
