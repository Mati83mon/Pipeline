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


def _pinned_range(package: str) -> str:
    """Return the version specifier for a package pinned as a range, not `==`."""
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line.startswith(package) and line[len(package) :].startswith((">", "<", "=", "~")):
            return line[len(package) :].strip()
    raise AssertionError(f"{package} is not pinned in requirements.txt")


def test_sdk_version_matches_pinned_gradio(front_matter):
    assert front_matter["sdk_version"] == _pinned_version("gradio"), (
        "README sdk_version and the gradio pin in requirements.txt must agree, "
        "or the Space builds one Gradio and installs another"
    )


def test_torch_pin_is_supported_by_zerogpu():
    assert _pinned_version("torch") in {"2.8.0", "2.9.1", "2.10.0", "2.11.0"}


def test_space_sync_workflow_is_gated_on_the_secret():
    """It must skip, not fail, on a checkout with no HF_TOKEN configured.

    Without the gate, merging this workflow turns CI red for anyone who has not
    opted in — and a red default branch is exactly the signal you cannot afford
    to teach people to ignore.
    """
    workflow = (REPO_ROOT / ".github/workflows/sync-space.yml").read_text(encoding="utf-8")
    assert "steps.gate.outputs.ready == 'true'" in workflow
    assert 'if [ -z "$HF_TOKEN" ]' in workflow
    # The token may only ever arrive through the secrets context.
    assert "hf_" not in workflow.lower().replace("hf_token", "")


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


def test_generation_failure_always_names_the_exception_type(tmp_path):
    """A bare KeyError renders as "'key'" — quoted, typeless and unactionable.

    That is exactly what reached the UI as `Error: 'GenerationError'`.
    """
    manager = _manager(tmp_path)
    for exc in (RuntimeError("boom"), KeyError("some_key"), ValueError("bad")):
        assert type(exc).__name__ in manager._explain_generation_failure(exc)


@pytest.mark.parametrize("exc", [KeyError("k"), AttributeError("a"), IndexError("i")])
def test_structural_failures_point_at_the_checkpoint_not_the_prompt(tmp_path, exc):
    explained = _manager(tmp_path)._explain_generation_failure(exc)
    assert "checkpoint" in explained
    assert "Space logs" in explained


def test_oom_message_still_carries_the_exception_type(tmp_path):
    explained = _manager(tmp_path)._explain_generation_failure(
        RuntimeError("CUDA out of memory")
    )
    assert "Lower the resolution" in explained
    assert "RuntimeError" in explained


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


def _install_fp8_experts(monkeypatch, overrides):
    """Install a stub transformers.integrations.finegrained_fp8 with FP8Experts."""
    import types

    class FP8Experts:
        _impl_tp_layer_overrides = overrides

    root = types.ModuleType("transformers")
    integrations = types.ModuleType("transformers.integrations")
    fp8 = types.ModuleType("transformers.integrations.finegrained_fp8")
    fp8.FP8Experts = FP8Experts
    integrations.finegrained_fp8 = fp8
    root.integrations = integrations
    monkeypatch.setitem(sys.modules, "transformers", root)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.finegrained_fp8", fp8)
    return FP8Experts


def test_fp8_tp_plan_patch_adds_the_missing_default(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    experts = _install_fp8_experts(monkeypatch, {"deepgemm_megamoe": {"a": "b"}})
    assert manager._patch_fp8_tp_plan() is not None
    assert experts._impl_tp_layer_overrides[None] == {}
    # The real key must survive untouched.
    assert experts._impl_tp_layer_overrides["deepgemm_megamoe"] == {"a": "b"}


def test_fp8_tp_plan_patch_is_idempotent(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    _install_fp8_experts(monkeypatch, {"deepgemm_megamoe": {}})
    assert manager._patch_fp8_tp_plan() is not None
    assert manager._patch_fp8_tp_plan() is None


def test_fp8_tp_plan_patch_is_a_noop_without_transformers(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert manager._patch_fp8_tp_plan() is None


def test_fp8_tp_plan_patch_fixes_the_upstream_crash(tmp_path, monkeypatch):
    """Reproduce transformers' update_tp_plan line, before and after the patch.

    This mirrors quantizer_finegrained_fp8.py:195 in transformers 5.16.1:

        layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)
        updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
    """
    manager = _manager(tmp_path)
    experts = _install_fp8_experts(monkeypatch, {"deepgemm_megamoe": {}})
    base_plan = {"layers.*.self_attn.q_proj.weight": "colwise"}
    impl = None  # transformers' default for _experts_implementation

    def upstream_line():
        overrides = experts._impl_tp_layer_overrides.get(impl)
        return {k: overrides.get(v, v) for k, v in base_plan.items()}

    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
        upstream_line()

    manager._patch_fp8_tp_plan()
    assert upstream_line() == base_plan  # unchanged plan, which is the intent


def _install_quantizer_utils(monkeypatch, buggy: bool):
    """Stub transformers.quantizers.quantizers_utils + integrations.finegrained_fp8."""
    import re
    import types

    def upstream_buggy(full_name, patterns=None):
        if patterns is None:
            return True
        return not any(
            re.match(f"{k}\\.", full_name)
            or re.match(f"{k}", full_name)  # the missing end anchor
            or full_name.endswith(k)
            for k in patterns
        )

    def upstream_fixed(full_name, patterns=None):
        if patterns is None:
            return True
        return not any(
            re.match(f"{k}\\.", full_name) or re.fullmatch(k, full_name) for k in patterns
        )

    fn = upstream_buggy if buggy else upstream_fixed

    root = types.ModuleType("transformers")
    quantizers = types.ModuleType("transformers.quantizers")
    utils = types.ModuleType("transformers.quantizers.quantizers_utils")
    integrations = types.ModuleType("transformers.integrations")
    fp8 = types.ModuleType("transformers.integrations.finegrained_fp8")
    utils.should_convert_module = fn
    fp8.should_convert_module = fn  # imported by value, as upstream does
    quantizers.quantizers_utils = utils
    integrations.finegrained_fp8 = fp8
    root.quantizers = quantizers
    root.integrations = integrations
    for name, mod in (
        ("transformers", root),
        ("transformers.quantizers", quantizers),
        ("transformers.quantizers.quantizers_utils", utils),
        ("transformers.integrations", integrations),
        ("transformers.integrations.finegrained_fp8", fp8),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return utils, fp8


GATE = "model.layers.0.mlp.gate"


def test_skip_matching_patch_fixes_gate_proj(tmp_path, monkeypatch):
    """gate_proj must be quantised even though gate is on the skip list."""
    manager = _manager(tmp_path)
    utils, fp8 = _install_quantizer_utils(monkeypatch, buggy=True)

    assert utils.should_convert_module(f"{GATE}_proj", [GATE]) is False  # the bug
    assert manager._patch_fp8_module_skip_matching() is not None
    assert utils.should_convert_module(f"{GATE}_proj", [GATE]) is True


def test_skip_matching_patch_preserves_intended_skips(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    utils, _ = _install_quantizer_utils(monkeypatch, buggy=True)
    manager._patch_fp8_module_skip_matching()

    assert utils.should_convert_module(GATE, [GATE]) is False           # router
    assert utils.should_convert_module("lm_head", ["lm_head"]) is False  # head
    assert utils.should_convert_module(f"{GATE}.weight", [GATE]) is False
    assert utils.should_convert_module("model.layers.0.mlp.up_proj", [GATE]) is True
    assert utils.should_convert_module("anything", None) is True


def test_skip_matching_patch_also_fixes_the_by_value_import(tmp_path, monkeypatch):
    """finegrained_fp8 imports the function by value, so it needs patching too."""
    manager = _manager(tmp_path)
    _, fp8 = _install_quantizer_utils(monkeypatch, buggy=True)
    manager._patch_fp8_module_skip_matching()
    assert fp8.should_convert_module(f"{GATE}_proj", [GATE]) is True


def test_skip_matching_patch_is_skipped_when_upstream_is_correct(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    _install_quantizer_utils(monkeypatch, buggy=False)
    assert manager._patch_fp8_module_skip_matching() is None


def test_skip_matching_patch_is_idempotent(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    _install_quantizer_utils(monkeypatch, buggy=True)
    assert manager._patch_fp8_module_skip_matching() is not None
    assert manager._patch_fp8_module_skip_matching() is None


def _install_kernels_probe(monkeypatch, available):
    """Stub transformers.utils.import_utils.is_kernels_available.

    `available=None` installs no probe at all, standing in for a transformers
    too old to have one.
    """
    import types

    root = types.ModuleType("transformers")
    utils = types.ModuleType("transformers.utils")
    import_utils = types.ModuleType("transformers.utils.import_utils")
    if available is not None:
        import_utils.is_kernels_available = lambda *a, **k: available
    utils.import_utils = import_utils
    root.utils = utils
    for name, mod in (
        ("transformers", root),
        ("transformers.utils", utils),
        ("transformers.utils.import_utils", import_utils),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def test_missing_fp8_kernels_are_reported(tmp_path, monkeypatch):
    """Without `kernels` the checkpoint loads and dies in the forward pass."""
    _install_kernels_probe(monkeypatch, available=False)
    reported = _manager(tmp_path)._check_fp8_kernels()
    assert reported is not None
    assert "kernels" in reported
    assert "0.16" in reported


def test_present_fp8_kernels_report_nothing(tmp_path, monkeypatch):
    _install_kernels_probe(monkeypatch, available=True)
    assert _manager(tmp_path)._check_fp8_kernels() is None


def test_fp8_kernel_probe_is_never_fatal(tmp_path, monkeypatch):
    """A transformers without the helper must not break loading."""
    _install_kernels_probe(monkeypatch, available=None)
    assert _manager(tmp_path)._check_fp8_kernels() is None


def test_fp8_load_surfaces_the_kernel_warning_to_the_ui(tmp_path, monkeypatch):
    """The notice must reach the user before a GPU allocation is spent."""
    from pipeline.models import ModelManager

    manager = _manager(tmp_path)
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers({"quant_method": "fp8"}, [])
    )
    monkeypatch.setattr(
        ModelManager, "_check_fp8_kernels", staticmethod(lambda: "kernels are missing")
    )
    manager._load_llm("some/model", {"dtype": "bfloat16", "multimodal": True})
    assert "kernels are missing" in manager.take_notices()


def test_requirements_pin_the_fp8_kernel_package():
    """transformers routes every FP8 matmul through `kernels`; it is not optional."""
    assert _pinned_range("kernels") == ">=0.16,<0.17"


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
# Reasoning modes
#
# The default text model's chat template reads `enable_thinking` and validates
# `reasoning_effort` itself:
#
#     {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
#     {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
#         {{- raise_exception('Unexpected reasoning effort ...') }}
#
# so an invalid preset is not a bad default, it is a hard failure at generation
# time. These tests pin the contract to what the template accepts.
# ---------------------------------------------------------------------------


TEMPLATE_EFFORTS = {"xhigh", "medium", "low"}


def test_thinking_modes_only_use_efforts_the_template_accepts():
    from pipeline.models import THINKING_MODES

    for name, kwargs in THINKING_MODES.items():
        effort = kwargs.get("reasoning_effort")
        assert effort is None or effort in TEMPLATE_EFFORTS, name
        assert isinstance(kwargs["enable_thinking"], bool), name


def test_thinking_off_sends_no_reasoning_effort():
    """With thinking disabled the template skips the effort block entirely."""
    from pipeline.models import THINKING_MODES

    assert THINKING_MODES["off"] == {"enable_thinking": False}


def test_every_preset_names_a_real_thinking_mode():
    import ui_components as ui

    from pipeline.models import THINKING_MODES

    for name, values in ui.TEXT_PRESETS.items():
        assert values["thinking"] in THINKING_MODES, name


def test_thinking_dropdown_offers_exactly_the_known_modes():
    import ui_components as ui

    from pipeline.models import THINKING_MODES

    assert {value for _, value in ui.THINKING_CHOICES} == set(THINKING_MODES)


@pytest.mark.parametrize(
    "given,expected",
    [("deep", "deep"), ("OFF", "off"), (" brief ", "brief"), (None, "standard"),
     ("nonsense", "standard"), ("", "standard")],
)
def test_thinking_key_is_normalised(tmp_path, given, expected):
    assert _manager(tmp_path)._thinking_key(given) == expected


def test_default_config_names_a_real_thinking_mode():
    from pipeline.models import THINKING_MODES

    config = load_config(REPO_ROOT / "config.json")
    assert config.llm["default_thinking"] in THINKING_MODES


def test_apply_template_forwards_the_reasoning_kwargs(tmp_path):
    seen = {}

    def apply(messages, **kwargs):
        seen.update(kwargs)
        return "prompt"

    result = _manager(tmp_path)._apply_template(
        apply, [], {"enable_thinking": True, "reasoning_effort": "low"}, tokenize=False
    )
    assert result == "prompt"
    assert seen == {"tokenize": False, "enable_thinking": True, "reasoning_effort": "low"}


def test_apply_template_retries_without_kwargs_a_template_rejects(tmp_path):
    """The fallback model's template has never heard of these flags."""
    calls = []

    def apply(messages, **kwargs):
        calls.append(kwargs)
        if "reasoning_effort" in kwargs:
            raise ValueError("Unexpected reasoning effort")
        return "plain prompt"

    result = _manager(tmp_path)._apply_template(
        apply, [], {"enable_thinking": True, "reasoning_effort": "low"}
    )
    assert result == "plain prompt"
    assert len(calls) == 2 and calls[1] == {}


def test_apply_template_does_not_swallow_a_real_failure(tmp_path):
    def apply(messages, **kwargs):
        raise RuntimeError("template is broken")

    with pytest.raises(RuntimeError, match="template is broken"):
        _manager(tmp_path)._apply_template(apply, [], {})


def test_build_llm_inputs_passes_the_selected_mode(tmp_path):
    seen = {}

    class _Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            seen.update(kwargs)
            return "text"

        def __call__(self, text, **kwargs):
            class _T:
                def to(self, device):
                    return self

            return {"input_ids": _T()}

    _manager(tmp_path)._build_llm_inputs(
        {"system_prompt": "s"}, None, _Tokenizer(), "hi", [], thinking="off"
    )
    assert seen["enable_thinking"] is False
    assert "reasoning_effort" not in seen


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------


class _FakeGenerated:
    """Minimal stand-in for the generated-token tensor slice."""

    def __init__(self, ids):
        self._ids = list(ids)
        self.shape = (len(self._ids),)

    def __getitem__(self, index):
        return self._ids[index]


class _FakeModel:
    def __init__(self, eos):
        self.generation_config = type("GC", (), {"eos_token_id": eos})()


def test_answer_that_ends_on_eos_is_not_truncated(tmp_path):
    manager = _manager(tmp_path)
    assert not manager._hit_token_ceiling(
        _FakeGenerated([1, 2, 151645]), _FakeModel(151645), None, None, 3
    )


def test_answer_that_fills_the_budget_without_eos_is_truncated(tmp_path):
    manager = _manager(tmp_path)
    assert manager._hit_token_ceiling(
        _FakeGenerated([1, 2, 3]), _FakeModel(151645), None, None, 3
    )


def test_short_answer_is_never_reported_as_truncated(tmp_path):
    """An early stop on a stop-string is a finished answer, not a cut-off one."""
    manager = _manager(tmp_path)
    assert not manager._hit_token_ceiling(
        _FakeGenerated([1, 2]), _FakeModel(151645), None, None, 512
    )


def test_eos_ids_are_collected_from_a_list(tmp_path):
    manager = _manager(tmp_path)
    assert not manager._hit_token_ceiling(
        _FakeGenerated([1, 2, 7]), _FakeModel([151643, 7]), None, None, 3
    )


def test_truncation_check_never_raises_on_an_odd_tensor(tmp_path):
    assert _manager(tmp_path)._hit_token_ceiling(
        object(), _FakeModel(1), None, None, 10
    ) is False


class _Decoder:
    """A tokenizer that treats `<think>` as special, so the plain decode drops it."""

    def __init__(self, raw):
        self._raw = raw

    def decode(self, ids, skip_special_tokens=True):
        return "" if skip_special_tokens else self._raw


class _PlainDecoder:
    """A tokenizer that does not.

    This checkpoint is the second kind: `tokenizer_config.json` registers
    `<think>` / `</think>` as added tokens with `"special": false`, so both
    decodes contain them. The advice must not depend on which kind it gets —
    an earlier version of this code was documented on the wrong assumption.
    """

    def __init__(self, raw):
        self._raw = raw

    def decode(self, ids, skip_special_tokens=True):
        return self._raw


@pytest.mark.parametrize("decoder_cls", [_Decoder, _PlainDecoder])
def test_advice_does_not_depend_on_how_think_tags_are_classified(
    tmp_path, decoder_cls
):
    manager = _manager(tmp_path)
    unfinished = manager._truncation_advice(
        _FakeGenerated([1]), decoder_cls("<think>still going"), "standard"
    )
    finished = manager._truncation_advice(
        _FakeGenerated([1]), decoder_cls("<think>done</think>an answer"), "standard"
    )
    assert "reasoning" in unfinished.lower()
    assert "reasoning" not in finished.lower()


def test_advice_points_at_reasoning_when_the_think_block_never_closed(tmp_path):
    advice = _manager(tmp_path)._truncation_advice(
        _FakeGenerated([1]), _Decoder("<think>still going"), "standard"
    )
    assert "reasoning" in advice.lower()
    assert "Brief" in advice or "Off" in advice


def test_advice_does_not_suggest_lowering_reasoning_when_it_is_already_off(tmp_path):
    advice = _manager(tmp_path)._truncation_advice(
        _FakeGenerated([1]), _Decoder("<think>odd"), "off"
    )
    assert "Max new tokens" in advice
    assert "Brief" not in advice


def test_advice_points_at_the_limit_when_the_answer_itself_was_cut(tmp_path):
    advice = _manager(tmp_path)._truncation_advice(
        _FakeGenerated([1]), _Decoder("<think>done</think>the answer"), "standard"
    )
    assert "Max new tokens" in advice
    assert "reasoning" not in advice.lower()


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


# ---------------------------------------------------------------------------
# The ZeroGPU worker boundary
#
# `spaces` ships a failed GPU task back to the parent as
# `ExceptionResult(traceback, error_cls, gradio_error)` and keeps the exception
# object itself only when it is a `gr.Error`; anything else is re-raised in the
# parent from its class name alone. That is how a `GenerationError` carrying a
# real diagnosis arrived in the UI as `Error: 'GenerationError'`.
# ---------------------------------------------------------------------------


def test_gpu_safe_passes_results_through(app_module):
    assert app_module._gpu_safe(lambda a, b: (a, b))(1, 2) == (1, 2)


def test_gpu_safe_preserves_the_wrapped_name(app_module):
    def _inner():
        return None

    assert app_module._gpu_safe(_inner).__name__ == "_inner"


def test_gpu_safe_converts_backend_errors_to_gradio_errors(app_module):
    """The regression: only a gr.Error survives the ZeroGPU worker boundary."""
    import gradio as gr

    from pipeline.models import GenerationError

    message = "ImportError: finegrained-fp8 kernel unavailable: kernels missing"

    @app_module._gpu_safe
    def _task():
        raise GenerationError(message)

    with pytest.raises(gr.Error) as caught:
        _task()
    assert message in str(caught.value)


def test_gpu_safe_does_not_rewrap_a_gradio_error(app_module):
    import gradio as gr

    original = gr.Error("already actionable")

    @app_module._gpu_safe
    def _task():
        raise original

    with pytest.raises(gr.Error) as caught:
        _task()
    assert caught.value is original


def test_converted_error_survives_the_process_boundary(app_module):
    """`spaces` pickles the gr.Error to the parent process — the message must live."""
    import pickle

    import gradio as gr

    from pipeline.models import GenerationError

    message = "the model failed internally (KeyError: 'router')"
    error = app_module._fail(GenerationError(message))
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, gr.Error)
    assert message in str(restored)


def test_fail_returns_a_gradio_error_unchanged(app_module):
    import gradio as gr

    error = gr.Error("Prompt cannot be empty.")
    assert app_module._fail(error) is error


def test_fail_names_the_type_of_an_unexpected_error(app_module):
    assert "TypeError" in str(app_module._fail(TypeError("bad argument")))


def test_reasoning_off_asks_for_a_shorter_gpu_allocation(app_module):
    """Quota is charged by the second; a no-reasoning run must not book for one."""
    args = ("prompt", None, None, 1000, 1.0, 0.95, 1)
    thinking_seconds = app_module._text_duration(*args, "standard")
    quick_seconds = app_module._text_duration(*args, "off")
    assert quick_seconds < thinking_seconds
    assert quick_seconds == pytest.approx(90 + 10.0 * 10)


def test_duration_defaults_to_the_thinking_budget(app_module):
    """An unknown or missing mode must never under-book the allocation."""
    args = ("prompt", None, None, 1000, 1.0, 0.95, 1)
    assert app_module._text_duration(*args) == app_module._text_duration(*args, "deep")


def test_preset_sets_all_four_controls():
    import ui_components as ui

    max_tokens, temperature, top_p, thinking = ui.apply_text_preset("Quick answer")
    assert (max_tokens, temperature, top_p, thinking) == (700, 0.7, 0.9, "off")


@pytest.mark.parametrize("name", ["Custom", "a preset that was renamed"])
def test_unknown_preset_leaves_the_dials_alone(name):
    """Selecting Custom must not reset values the user just tuned by hand.

    `gr.update()` with no `value` is Gradio's "change nothing" sentinel; a bare
    value here would silently overwrite the sliders instead.
    """
    import ui_components as ui

    updates = ui.apply_text_preset(name)
    assert len(updates) == 4
    for update in updates:
        assert update == {"__type__": "update"}


def test_continue_without_a_previous_answer_is_rejected(app_module):
    import gradio as gr

    with pytest.raises(gr.Error, match="Nothing to continue"):
        app_module.handle_continue(
            "carry on", "", None, None, 512, 1.0, 0.95, 1, True, "standard"
        )


def test_continuation_prompt_carries_the_previous_answer(app_module):
    previous = "The first half of a long answer."
    built = app_module.CONTINUATION_TEMPLATE.format(previous=previous, prompt="Explain X")
    assert previous in built
    assert "Explain X" in built
    assert "not repeat" in built.lower()


def test_continuation_is_not_rejected_for_exceeding_the_typed_prompt_limit(
    app_module, monkeypatch
):
    """A long answer is exactly what is worth continuing; the cap guards typing."""
    seen = {}

    def _capture(prompt, *args, **kwargs):
        seen["prompt"] = prompt
        return "", "", *("", ""), ""

    monkeypatch.setattr(app_module, "_run_text", _capture)
    previous = "x" * (app_module.MAX_PROMPT_CHARS * 2)
    app_module.handle_continue(
        "go on", previous, None, None, 512, 1.0, 0.95, 1, True, "standard"
    )
    assert previous in seen["prompt"]


@pytest.mark.parametrize(
    "task,method,arity",
    [
        ("_gpu_image", "generate_image", 7),
        ("_gpu_video", "generate_video", 10),
        ("_gpu_text", "generate_text", 8),
    ],
)
def test_every_gpu_task_converts_before_the_boundary(
    app_module, monkeypatch, task, method, arity
):
    """Guards the decorator itself: drop `@_gpu_safe` and the message is lost."""
    import gradio as gr

    from pipeline.models import GenerationError

    def _boom(*args, **kwargs):
        raise GenerationError("a diagnosis worth keeping")

    monkeypatch.setattr(app_module.MANAGER, method, _boom)
    with pytest.raises(gr.Error) as caught:
        getattr(app_module, task)(*[None] * arity)
    assert "a diagnosis worth keeping" in str(caught.value)


# ---------------------------------------------------------------------------
# Remote video engine — MiniMax-H3 on another Space
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Stands in for `gradio_client.Client`, recording how it was built and called."""

    instances: list = []

    def __init__(self, src, **kwargs):
        self.src = src
        self.kwargs = kwargs
        self.calls = []
        self.result = (None, "remote report", "")
        _RecordingClient.instances.append(self)

    def predict(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _fake_gradio_client(monkeypatch, video_path=None):
    """Install a stub `gradio_client` and return the module.

    `remote.py` imports it inside the methods that need it, which is what makes
    this possible without the real package — and is also why the production
    import failure has its own message.
    """
    import types

    _RecordingClient.instances = []
    module = types.ModuleType("gradio_client")
    module.Client = _RecordingClient
    module.handle_file = lambda path: {"path": path, "meta": {"_type": "gradio.FileData"}}
    monkeypatch.setitem(sys.modules, "gradio_client", module)
    if video_path is not None:
        _RecordingClient.result_path = str(video_path)
    return module


@pytest.fixture
def remote_engine(tmp_path):
    from pipeline.remote import RemoteVideoEngine

    config = load_config(REPO_ROOT / "config.json")
    return RemoteVideoEngine(config.video_remote, output_dir=str(tmp_path / "outputs"))


# -- the ported cost model -------------------------------------------------


def _upstream_duration(image, last_image, height, width, num_frames, steps):
    """`get_duration` transcribed from the remote Space's own `app.py`.

    An independent copy on purpose. It is the control for our port: edit the
    coefficients in `pipeline.remote` and the grid test below fails, which is
    the only way a silently wrong quota estimate gets caught — the UI prints
    the number with confidence either way.
    """
    frames_per_chunk, latents_per_chunk = 17, 5
    dur_b, dur_c = 1.1745e-4, 3.8396e-9
    decode_base, decode_per_canvas = 15, 15
    default_canvas_pixels = 960 * 544 * 124
    placement_allowance, pad = 12, 10

    height, width, num_frames, steps = int(height), int(width), int(num_frames), int(steps)
    latent_frames = (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2
    patches = (height // 32) * (width // 32)
    rows = latent_frames * patches + (int(image is not None) + int(last_image is not None)) * patches
    denoise = steps * (dur_b * rows + dur_c * rows**2)
    decode = decode_base + decode_per_canvas * (height * width * num_frames) / default_canvas_pixels
    return max(60, int(denoise + decode) + placement_allowance + pad)


def test_ported_cost_model_matches_the_remote_formula():
    from pipeline.remote import CANVASES, estimate_gpu_seconds, snap_frames

    for label, (height, width) in CANVASES.items():
        for seconds in (2, 2.5, 5, 10, 14):
            for steps in (4, 6, 28):
                for keyframes in (0, 1, 2):
                    expected = _upstream_duration(
                        object() if keyframes >= 1 else None,
                        object() if keyframes >= 2 else None,
                        height,
                        width,
                        snap_frames(seconds),
                        steps,
                    )
                    assert (
                        estimate_gpu_seconds(label, seconds, steps, keyframes) == expected
                    ), f"{label} {seconds}s {steps} steps {keyframes} keyframes"


@pytest.mark.parametrize(
    "canvas,seconds,steps,keyframes,expected",
    [
        ("960x544 · 16:9 fast", 5, 6, 0, 73),
        # The floor: nothing is booked for less than 60 s.
        ("960x544 · 16:9 fast", 2, 4, 0, 60),
        ("768x1344 · 9:16 full", 10, 6, 1, 271),
        # The longest clip this tab can ask for, at the widest canvas.
        ("1344x768 · 16:9 full", 14, 6, 1, 440),
        # Undistilled base model: 28 steps instead of 6, and it shows.
        ("1536x672 · 21:9 full", 10, 28, 0, 900),
    ],
)
def test_cost_model_absolute_values_are_pinned(canvas, seconds, steps, keyframes, expected):
    """Pinned so editing both our port and the transcription still fails."""
    from pipeline.remote import estimate_gpu_seconds

    assert estimate_gpu_seconds(canvas, seconds, steps, keyframes) == expected


@pytest.mark.parametrize(
    "seconds,frames",
    [(2, 56), (5, 124), (10, 243), (13, 328), (13.5, 328), (14, 345)],
)
def test_frame_snapping_matches_the_video_vae(seconds, frames):
    """The decodable counts are `17n + 5` at 24 fps, nothing else."""
    from pipeline.remote import snap_frames

    assert snap_frames(seconds) == frames
    assert frames % 17 == 5


def test_fourteen_seconds_is_the_real_ceiling(remote_engine):
    """Why `max_duration` is 14 and not 15.

    15 s is 360 frames, which snaps to 362 — 15.083 s — past the duration the
    pipeline accepts, so the request is refused rather than truncated. 14 s
    snaps to 345 (14.375 s) and lands. A slider that offered 15 would hand the
    user a guaranteed failure.
    """
    from pipeline.remote import snap_frames

    low, high = remote_engine.duration_bounds()
    assert (low, high) == (2.0, 14.0)
    assert snap_frames(high) / 24 <= 15.0
    assert snap_frames(high + 0.5) / 24 > 15.0


# -- config --------------------------------------------------------------


def test_shipped_config_enables_the_remote_engine():
    config = load_config(REPO_ROOT / "config.json")
    assert config.remote_video_enabled
    block = config.video_remote
    assert block["space_id"].count("/") == 1
    assert block["max_duration"] == 14
    assert block["default_lora"] in block["loras"]


def test_config_canvas_table_matches_the_module():
    """Two copies of the same labels; neither may drift from the other.

    The label is matched by string on the far side, so a typo here is a request
    that fails inside a conditioner Space the user has never heard of.
    """
    from pipeline.remote import CANVASES

    config = load_config(REPO_ROOT / "config.json")
    assert config.video_remote["canvases"] == list(CANVASES)
    # The separator upstream uses, and the one a lookup will be matched against.
    assert all("·" in label for label in CANVASES)


def _remote_config(tmp_path, **overrides):
    raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    raw["video"]["remote_engine"].update(overrides)
    broken = tmp_path / "config.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    return broken


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"space_id": "no-slash"}, "must be 'owner/name'"),
        ({"space_id": "owner/"}, "must be 'owner/name'"),
        ({"canvases": []}, "non-empty list"),
        ({"default_canvas": "640x480 · nope"}, "is not in"),
        ({"min_duration": 20}, "exceeds 'max_duration'"),
        ({"default_duration": 99}, "outside"),
        ({"default_lora": "nonexistent"}, "is not in"),
        ({"loras": {"larry": "six"}}, "step count"),
    ],
)
def test_remote_engine_config_is_validated(tmp_path, overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_remote_config(tmp_path, **overrides))


def test_disabled_remote_engine_skips_validation(tmp_path):
    """A block that is off must not be able to stop the Space from booting."""
    path = _remote_config(tmp_path, enabled=False, space_id="broken", canvases=[])
    config = load_config(path)
    assert not config.remote_video_enabled


def test_absent_remote_engine_is_fine(tmp_path):
    raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    del raw["video"]["remote_engine"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(path)
    assert config.video_remote == {}
    assert not config.remote_video_enabled


# -- the call itself -----------------------------------------------------


def _generate(engine, tmp_path, **kwargs):
    produced = tmp_path / "remote.mp4"
    produced.write_bytes(b"\x00mp4")
    _RecordingClient.instances = []
    defaults = dict(prompt="a cat on a skateboard", duration=5, steps=6, seed=7)
    defaults.update(kwargs)

    original_init = _RecordingClient.__init__

    def init(self, src, **client_kwargs):
        original_init(self, src, **client_kwargs)
        self.result = (str(produced), "1344x768 · 345 frames", "")

    _RecordingClient.__init__ = init
    try:
        return engine.generate(**defaults)
    finally:
        _RecordingClient.__init__ = original_init


def test_ip_token_is_forwarded_so_the_caller_pays(remote_engine, tmp_path, monkeypatch):
    """The one header this integration cannot work without.

    ZeroGPU bills a booking to whatever `x-ip-token` identifies. Drop it and
    both bookings — the denoiser's and its conditioner's — land on this Space's
    pod IP, where an unattributed caller may hold 120 credits and an xlarge
    second costs two. A render of any useful length can never be granted one,
    so the symptom is not a bigger bill but a queue that never clears.
    """
    _fake_gradio_client(monkeypatch)
    _generate(remote_engine, tmp_path, ip_token="tok-abc")
    call = _RecordingClient.instances[0].calls[0]
    assert call["headers"] == {"x-ip-token": "tok-abc"}


def test_no_ip_token_sends_no_header(remote_engine, tmp_path, monkeypatch):
    _fake_gradio_client(monkeypatch)
    _generate(remote_engine, tmp_path, ip_token=None)
    assert _RecordingClient.instances[0].calls[0]["headers"] is None


def test_hf_token_is_not_sent_to_a_space_we_do_not_own(
    remote_engine, tmp_path, monkeypatch
):
    """A read credential for a gated model must not leak to a third party.

    The default `space_id` belongs to someone else. Attaching `HF_TOKEN` to its
    requests would hand our token to whoever runs it, and buy nothing: the
    default target is public.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    _fake_gradio_client(monkeypatch)
    _generate(remote_engine, tmp_path)
    built = _RecordingClient.instances[0]
    assert built.kwargs["token"] is None
    assert "hf_secret_value" not in repr(built.kwargs)


def test_hf_token_is_sent_only_when_the_config_opts_in(tmp_path, monkeypatch):
    from pipeline.remote import RemoteVideoEngine

    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    _fake_gradio_client(monkeypatch)
    config = load_config(REPO_ROOT / "config.json")
    block = dict(config.video_remote)
    block["send_hf_token"] = True
    engine = RemoteVideoEngine(block, output_dir=str(tmp_path / "outputs"))
    _generate(engine, tmp_path)
    assert _RecordingClient.instances[0].kwargs["token"] == "hf_secret_value"


def test_generate_sends_the_parameters_the_endpoint_declares(
    remote_engine, tmp_path, monkeypatch
):
    _fake_gradio_client(monkeypatch)
    _generate(
        remote_engine,
        tmp_path,
        canvas="1344x768 · 16:9 full",
        duration=10,
        steps=4,
        lora="lightx",
        upsample=True,
        seed=99,
    )
    call = _RecordingClient.instances[0].calls[0]
    assert call["api_name"] == "/generate"
    assert call["canvas"] == "1344x768 · 16:9 full"
    assert call["duration"] == 10.0
    assert call["steps"] == 4
    assert call["lora"] == "lightx"
    assert call["upsample"] is True
    assert call["seed"] == 99
    # Every input the remote signature declares, and nothing it does not.
    assert set(call) == {
        "prompt", "image_path", "last_image_path", "canvas", "duration",
        "steps", "seed", "upsample", "lora", "api_name", "headers",
    }


def test_keyframes_travel_as_file_handles(remote_engine, tmp_path, monkeypatch):
    _fake_gradio_client(monkeypatch)
    first = tmp_path / "first.png"
    first.write_bytes(b"\x89PNG")
    _generate(remote_engine, tmp_path, image=str(first))
    call = _RecordingClient.instances[0].calls[0]
    assert call["image_path"]["path"] == str(first)
    assert call["last_image_path"] is None


def test_output_is_adopted_into_this_spaces_output_dir(
    remote_engine, tmp_path, monkeypatch
):
    """Otherwise it is the one render History never shows and pruning never caps."""
    _fake_gradio_client(monkeypatch)
    path, _report, _refined, _elapsed = _generate(remote_engine, tmp_path, seed=4242)
    assert Path(path).parent == Path(remote_engine.output_dir)
    assert Path(path).name.startswith("video_h3_")
    assert Path(path).name.endswith("_4242.mp4")
    assert Path(path).read_bytes() == b"\x00mp4"


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"canvas": "640x480 · invented"}, "not a canvas"),
        ({"duration": 15}, "checkpoint's ceiling"),
        ({"duration": 0.5}, "between"),
    ],
)
def test_rejected_requests_never_reach_the_network(
    remote_engine, tmp_path, monkeypatch, overrides, message
):
    """Validate locally, so a doomed request costs nothing and says why."""
    from pipeline.remote import RemoteEngineError

    _fake_gradio_client(monkeypatch)
    with pytest.raises(RemoteEngineError, match=message):
        _generate(remote_engine, tmp_path, **overrides)
    assert _RecordingClient.instances == []


def test_disabled_engine_refuses_with_an_actionable_message(tmp_path):
    from pipeline.remote import RemoteEngineError, RemoteVideoEngine

    engine = RemoteVideoEngine({"enabled": False}, output_dir=str(tmp_path))
    with pytest.raises(RemoteEngineError, match="config.json"):
        engine.generate(prompt="x")


@pytest.mark.parametrize(
    "result,expected",
    [
        (("/v.mp4", "report", "refined"), ("/v.mp4", "report", "refined")),
        (["/v.mp4", "report"], ("/v.mp4", "report", "")),
        ({"path": "/v.mp4"}, ("/v.mp4", "", "")),
        ("/v.mp4", ("/v.mp4", "", "")),
        (("/v.mp4", "r", "x", "a fourth element"), ("/v.mp4", "r", "x")),
    ],
)
def test_unpack_tolerates_the_shapes_the_endpoint_can_return(result, expected):
    from pipeline.remote import RemoteVideoEngine

    assert RemoteVideoEngine._unpack(result) == expected


def test_unpack_rejects_an_empty_result():
    from pipeline.remote import RemoteEngineError, RemoteVideoEngine

    with pytest.raises(RemoteEngineError, match="no video"):
        RemoteVideoEngine._unpack(("", "", ""))


def test_remote_module_books_no_gpu():
    """The premise of the whole design: this side allocates nothing.

    A `@spaces.GPU` here would hold a local card for the length of a network
    call to a Space that is itself already holding one — paying twice for one
    render, on a quota charged by the second.

    Checked against the parse tree rather than the text, because the module
    explains this in prose and a substring search cannot tell the explanation
    from the mistake.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "pipeline/remote.py").read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            if node.module.endswith("runtime"):
                imported.update(alias.name for alias in node.names)
    assert "spaces" not in imported
    assert "gpu_task" not in imported

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                rendered = ast.unparse(decorator)
                assert "spaces" not in rendered and "gpu" not in rendered.lower(), (
                    f"{node.name} is decorated with {rendered}"
                )


# -- handler and UI wiring ------------------------------------------------


def test_remote_handler_forwards_the_request_token(app_module, tmp_path, monkeypatch):
    class _Request:
        headers = {"x-ip-token": "tok-from-request"}

    seen = {}

    def _generate_stub(**kwargs):
        seen.update(kwargs)
        produced = tmp_path / "out.mp4"
        produced.write_bytes(b"\x00")
        return str(produced), "remote says so", "", 12.5

    monkeypatch.setattr(app_module.REMOTE_VIDEO, "generate", _generate_stub)
    monkeypatch.setattr(
        app_module.MANAGER, "logger", GenerationLogger(str(tmp_path / "log.jsonl"))
    )
    path, info, *_ = app_module.handle_video_remote(
        "a prompt", None, None, "960x544 · 16:9 fast", 5, 6, "larry", False,
        7, False, _Request(),
    )
    assert seen["ip_token"] == "tok-from-request"
    assert "remote says so" in info
    assert Path(path).exists()


def test_remote_handler_logs_to_history(app_module, tmp_path, monkeypatch):
    """A remote render is a run like any other, or History lies about the Space."""
    logger = GenerationLogger(str(tmp_path / "log.jsonl"))
    monkeypatch.setattr(app_module.MANAGER, "logger", logger)
    monkeypatch.setattr(app_module.MANAGER, "output_dir", str(tmp_path))

    def _generate_stub(**kwargs):
        produced = tmp_path / "out.mp4"
        produced.write_bytes(b"\x00")
        return str(produced), "", "", 3.0

    monkeypatch.setattr(app_module.REMOTE_VIDEO, "generate", _generate_stub)
    app_module.handle_video_remote(
        "a prompt", None, None, "960x544 · 16:9 fast", 5, 6, "larry", False,
        7, False, None,
    )
    entry = logger.history(limit=1)[0]
    assert entry["type"] == "video"
    assert entry["params"]["engine"] == "minimax_h3"
    assert entry["params"]["attributed"] is False
    assert entry["model_id"] == app_module.REMOTE_VIDEO.space_id


def test_remote_handler_is_not_a_gpu_task(app_module):
    """`gpu_task` wraps with functools.wraps, so a wrapped function shows it."""
    assert not hasattr(app_module.handle_video_remote, "__wrapped__")
    for name in ("_gpu_image", "_gpu_video", "_gpu_text"):
        assert hasattr(getattr(app_module, name), "__wrapped__")


def test_remote_event_gets_its_own_concurrency_lane():
    """A six-minute network call must not sit in the one-model-resident queue.

    `default_concurrency_limit=1` is correct for the local tabs and wrong here:
    without a separate lane, one MiniMax render freezes image and text
    generation for every user for its whole duration.
    """
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    h3_click = source.split('video_ui["h3_button"].click(')[1].split("))")[0]
    assert 'concurrency_id="remote_video"' in h3_click
    assert "concurrency_limit" in h3_click


def test_video_tab_exposes_both_engines(app_module):
    import gradio as gr

    config = load_config(REPO_ROOT / "config.json")
    with gr.Blocks():
        built = app_module.ui.create_video_tab(config.video)
    assert built["remote_enabled"] is True
    for key in ("h3_canvas", "h3_duration", "h3_steps", "h3_lora", "h3_upsample"):
        assert key in built
    # The local controls survive untouched.
    for key in ("negative", "width", "height", "frames", "fps", "guidance"):
        assert key in built


def test_video_tab_without_the_remote_engine_is_the_old_tab(app_module):
    import gradio as gr

    config = load_config(REPO_ROOT / "config.json")
    block = dict(config.video)
    block["remote_engine"] = {"enabled": False}
    with gr.Blocks():
        built = app_module.ui.create_video_tab(block)
    assert built["remote_enabled"] is False
    # The group and the button are structural and always built; what must not
    # exist is any remote *control*, because app.py wires them by these names
    # and a half-built tab would fail at startup rather than at click time.
    for key in ("h3_canvas", "h3_duration", "h3_steps", "h3_lora", "h3_upsample",
                "h3_first_image", "h3_last_image", "h3_cost"):
        assert key not in built


def test_engine_toggle_swaps_groups_and_buttons(app_module):
    ltx_group, h3_group, ltx_button, h3_button = app_module.ui.toggle_video_engine(
        app_module.ui.ENGINE_H3
    )
    assert h3_group["visible"] and h3_button["visible"]
    assert not ltx_group["visible"] and not ltx_button["visible"]

    ltx_group, h3_group, ltx_button, h3_button = app_module.ui.toggle_video_engine(
        app_module.ui.ENGINE_LTX
    )
    assert ltx_group["visible"] and ltx_button["visible"]
    assert not h3_group["visible"] and not h3_button["visible"]


def test_request_readout_states_the_real_duration_and_cost(app_module):
    """The slider says 14; the model renders 345 frames. Show what will happen."""
    text = app_module.ui.describe_remote_request("1344x768 · 16:9 full", 14, 6, 0)
    assert "345 frames" in text
    assert "14.38 s of video" in text
    assert "xlarge" in text


def test_readme_documents_the_configured_remote_space(front_matter):
    """Drift this project has already been bitten by: config says one thing,
    README another. The space_id is the one value an operator will go looking
    for in the docs before they look in the JSON."""
    config = load_config(REPO_ROOT / "config.json")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert config.video_remote["space_id"] in readme
    # The model the remote Space runs belongs in the front matter like any other.
    assert "MiniMaxAI/MiniMax-H3" in set(front_matter.get("models", []))


def test_readme_states_the_real_duration_ceiling():
    """A README promising 15 s would send people straight into the refusal."""
    config = load_config(REPO_ROOT / "config.json")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{config.video_remote['max_duration']} seconds is the ceiling" in readme
