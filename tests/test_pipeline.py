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
