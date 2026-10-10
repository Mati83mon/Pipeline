"""Configuration loading and validation.

The original version of this project read ``config.json`` with a bare
``json.load`` at import time in two different modules, so a missing key only
surfaced as a ``KeyError`` deep inside a click handler. Here the schema is
validated once, up front, and every consumer gets a typed accessor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.environ.get("PIPELINE_CONFIG", "config.json")

# Keys that must exist for each module, with the type they must be coercible to.
_REQUIRED: Dict[str, Dict[str, type]] = {
    "image": {
        "model_id": str,
        "pipeline_class": str,
        "default_steps": int,
        "default_guidance": float,
        "default_width": int,
        "default_height": int,
        "min_steps": int,
        "max_steps": int,
        "min_guidance": float,
        "max_guidance": float,
    },
    "video": {
        "model_id": str,
        "pipeline_class": str,
        "default_frames": int,
        "default_fps": int,
        "default_steps": int,
        "min_frames": int,
        "max_frames": int,
        "min_fps": int,
        "max_fps": int,
    },
    "llm": {
        "model_id": str,
        "default_max_tokens": int,
        "default_temperature": float,
        "default_top_p": float,
        "min_tokens": int,
        "max_tokens": int,
    },
    "system": {
        "output_dir": str,
        "log_file": str,
        "history_limit": int,
    },
}


class ConfigError(ValueError):
    """Raised when config.json is missing or structurally invalid."""


@dataclass(frozen=True)
class AppConfig:
    """Validated view over ``config.json``."""

    raw: Dict[str, Any]
    path: Path = field(default=Path(DEFAULT_CONFIG_PATH))

    def section(self, name: str) -> Dict[str, Any]:
        try:
            return self.raw[name]
        except KeyError as exc:  # pragma: no cover - guarded by validation
            raise ConfigError(f"config.json has no '{name}' section") from exc

    @property
    def image(self) -> Dict[str, Any]:
        return self.section("image")

    @property
    def video(self) -> Dict[str, Any]:
        return self.section("video")

    @property
    def llm(self) -> Dict[str, Any]:
        return self.section("llm")

    @property
    def video_remote(self) -> Dict[str, Any]:
        """The optional remote video engine block. ``{}`` when absent."""
        block = self.video.get("remote_engine")
        return block if isinstance(block, dict) else {}

    @property
    def remote_video_enabled(self) -> bool:
        block = self.video_remote
        return bool(block.get("enabled")) and bool(str(block.get("space_id", "")).strip())

    @property
    def system(self) -> Dict[str, Any]:
        return self.section("system")

    def is_enabled(self, module: str) -> bool:
        return bool(self.section(module).get("enabled", True))

    @property
    def enabled_modules(self) -> list[str]:
        return [m for m in ("image", "video", "llm") if self.is_enabled(m)]


def _validate(raw: Dict[str, Any], path: str) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a JSON object")

    for section, fields_ in _REQUIRED.items():
        if section not in raw:
            raise ConfigError(f"{path}: missing required section '{section}'")
        if not isinstance(raw[section], dict):
            raise ConfigError(f"{path}: section '{section}' must be an object")
        for key, kind in fields_.items():
            if key not in raw[section]:
                raise ConfigError(f"{path}: '{section}.{key}' is required")
            value = raw[section][key]
            if kind is float and isinstance(value, int):
                continue
            if not isinstance(value, kind):
                raise ConfigError(
                    f"{path}: '{section}.{key}' must be {kind.__name__}, got "
                    f"{type(value).__name__}"
                )

    for section in ("image", "video", "llm"):
        block = raw[section]
        lo, hi = block.get("min_steps"), block.get("max_steps")
        if lo is not None and hi is not None and lo > hi:
            raise ConfigError(f"{path}: '{section}.min_steps' exceeds 'max_steps'")

    _validate_remote_engine(raw["video"].get("remote_engine"), path)


def _validate_remote_engine(block: Any, path: str) -> None:
    """Check the remote video engine block, when one is configured and on.

    The whole reason this file exists is that a typo should not surface as a
    ``KeyError`` at click time, and this block is the worst candidate for that:
    a wrong canvas label or a missing ``space_id`` does not fail here, it fails
    minutes later as an opaque error raised by a Space the user does not run.
    """
    where = "video.remote_engine"
    if block is None:
        return
    if not isinstance(block, dict):
        raise ConfigError(f"{path}: '{where}' must be an object")
    if not block.get("enabled"):
        return

    space_id = block.get("space_id")
    if not isinstance(space_id, str) or space_id.count("/") != 1 or not all(
        part.strip() for part in space_id.split("/")
    ):
        raise ConfigError(
            f"{path}: '{where}.space_id' must be 'owner/name', got {space_id!r}"
        )

    canvases = block.get("canvases")
    if not isinstance(canvases, list) or not canvases:
        raise ConfigError(f"{path}: '{where}.canvases' must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in canvases):
        raise ConfigError(f"{path}: '{where}.canvases' must hold non-empty strings")

    default_canvas = block.get("default_canvas")
    if default_canvas is not None and default_canvas not in canvases:
        raise ConfigError(
            f"{path}: '{where}.default_canvas' ({default_canvas!r}) is not in "
            f"'{where}.canvases'"
        )

    lo, hi = block.get("min_duration"), block.get("max_duration")
    for key, value in (("min_duration", lo), ("max_duration", hi)):
        if value is not None and not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: '{where}.{key}' must be a number")
    if lo is not None and hi is not None and lo > hi:
        raise ConfigError(f"{path}: '{where}.min_duration' exceeds 'max_duration'")

    default_duration = block.get("default_duration")
    if default_duration is not None:
        if not isinstance(default_duration, (int, float)):
            raise ConfigError(f"{path}: '{where}.default_duration' must be a number")
        if lo is not None and hi is not None and not lo <= default_duration <= hi:
            raise ConfigError(
                f"{path}: '{where}.default_duration' ({default_duration}) is "
                f"outside {lo}..{hi}"
            )

    loras = block.get("loras")
    if loras is not None:
        if not isinstance(loras, dict) or not loras:
            raise ConfigError(f"{path}: '{where}.loras' must be a non-empty object")
        if not all(isinstance(v, int) for v in loras.values()):
            raise ConfigError(
                f"{path}: '{where}.loras' must map a name to its step count (int)"
            )
        default_lora = block.get("default_lora")
        if default_lora is not None and default_lora not in loras:
            raise ConfigError(
                f"{path}: '{where}.default_lora' ({default_lora!r}) is not in "
                f"'{where}.loras'"
            )


def load_config(path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read and validate the config file.

    Resolves relative paths against the repository root so the app behaves the
    same whether it is launched from the repo root or from elsewhere.
    """
    candidate = Path(path)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = Path(__file__).resolve().parent.parent / candidate

    try:
        text = candidate.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {candidate}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{candidate}: invalid JSON ({exc})") from exc

    _validate(raw, str(candidate))
    return AppConfig(raw=raw, path=candidate)
