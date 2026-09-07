"""Runtime environment helpers: ZeroGPU, VRAM accounting and disk budgeting.

Three things here that the naive version of this app got wrong:

1. **ZeroGPU.** On ZeroGPU Spaces there is no GPU outside a ``@spaces.GPU``
   function, so a pipeline that calls ``.to("cuda")`` at request time without
   the decorator silently runs on an emulated device and then fails. The
   ``gpu_task`` decorator here is a no-op when ``spaces`` is unavailable, so the
   same code runs on ZeroGPU, on a dedicated GPU Space, in Docker and on CPU.

2. **CUDA probing.** ``torch.cuda.get_device_properties(0)`` raises when no
   device is present, and under ZeroGPU's CUDA emulation the reported numbers
   outside a GPU task are not meaningful. Every probe here is guarded.

3. **Disk.** A Space has a modest ephemeral disk, while the three model
   families used by this app total roughly 85 GB. Downloading a second model
   without freeing the first fills the disk and the Space dies with an opaque
   ``OSError``. ``ensure_disk_budget`` evicts other snapshots from the Hub cache
   first.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
from typing import Any, Callable, Iterable, Optional

LOGGER = logging.getLogger("pipeline.runtime")

# ---------------------------------------------------------------------------
# ZeroGPU integration
# ---------------------------------------------------------------------------

# `spaces` must be imported before torch when it is present; app.py does that.
try:  # pragma: no cover - depends on deployment target
    import spaces as _spaces

    HAS_SPACES = True
except Exception:  # noqa: BLE001 - any import failure means "not on Spaces"
    _spaces = None
    HAS_SPACES = False

IS_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "true", "yes"}


def gpu_task(duration: int | Callable[..., float] = 60, size: Optional[str] = None):
    """Decorate a function that needs a real GPU.

    On ZeroGPU this maps to ``spaces.GPU``; everywhere else it is transparent.
    ``duration`` may be a callable receiving the same arguments as the wrapped
    function, which lets a long video render ask for more wall-clock than a
    quick image.
    """

    def decorator(fn):
        if not HAS_SPACES:
            return fn
        kwargs: dict[str, Any] = {"duration": duration}
        if size:
            kwargs["size"] = size
        try:
            return _spaces.GPU(**kwargs)(fn)
        except TypeError:
            # Older `spaces` releases do not accept `size`.
            return _spaces.GPU(duration=duration)(fn)

    return decorator


def hf_token() -> Optional[str]:
    """Token used for gated repos, read from the Space secret or local env."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    return None


# ---------------------------------------------------------------------------
# Device / VRAM
# ---------------------------------------------------------------------------


def _torch():
    import torch  # imported lazily so config-only tests need no torch

    return torch


def cuda_available() -> bool:
    try:
        torch = _torch()
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:  # noqa: BLE001
        return False


def preferred_device() -> str:
    """Device generated tensors should end up on.

    Under ZeroGPU this is always ``cuda``: outside a GPU task PyTorch runs in
    emulation mode and accepts the placement, inside one it is a real device.
    """
    if IS_ZEROGPU:
        return "cuda"
    return "cuda" if cuda_available() else "cpu"


def supports_bfloat16() -> bool:
    if IS_ZEROGPU:
        return True  # Blackwell backing hardware
    try:
        torch = _torch()
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        return False


def resolve_dtype(name: str):
    """Map a config dtype string onto a torch dtype, degrading safely.

    ``float16`` is deliberately *not* the default: the T5 text encoders used by
    both Chroma and LTX overflow to NaN in fp16, which shows up as black images
    rather than as an error.
    """
    torch = _torch()
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = table.get(str(name).lower(), torch.bfloat16)
    if dtype is torch.bfloat16 and not supports_bfloat16():
        return torch.float32 if not cuda_available() else torch.float16
    if dtype is not torch.float32 and not cuda_available() and not IS_ZEROGPU:
        return torch.float32  # half precision on CPU is pathologically slow
    return dtype


def free_vram() -> None:
    gc.collect()
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def vram_status() -> str:
    if not cuda_available():
        return "ZeroGPU (on demand)" if IS_ZEROGPU else "CPU only"
    try:
        torch = _torch()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{allocated:.1f} / {total:.0f} GB used · {reserved:.1f} GB reserved"
    except Exception as exc:  # noqa: BLE001
        return f"VRAM unavailable ({type(exc).__name__})"


# ---------------------------------------------------------------------------
# Disk budgeting
# ---------------------------------------------------------------------------


def hub_cache_dir() -> str:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return explicit
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def free_disk_gb(path: Optional[str] = None) -> float:
    target = path or hub_cache_dir()
    probe = target
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or "/").free / 1024**3
    except Exception:  # noqa: BLE001
        return float("inf")


def disk_status() -> str:
    free = free_disk_gb()
    if free == float("inf"):
        return "disk usage unknown"
    return f"{free:.1f} GB free"


def _cached_repos() -> list[Any]:
    try:
        from huggingface_hub import scan_cache_dir

        return list(scan_cache_dir(cache_dir=hub_cache_dir()).repos)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("cache scan failed: %s", exc)
        return []


def ensure_disk_budget(
    needed_gb: float,
    keep: Iterable[str],
    headroom_gb: float = 10.0,
) -> list[str]:
    """Evict cached model snapshots until ``needed_gb`` + headroom is free.

    ``keep`` lists repo ids that must survive (the model about to be used, and
    anything currently resident). Returns the repo ids that were deleted so the
    caller can tell the user why the next run re-downloads.
    """
    keep_set = {k for k in keep if k}
    required = float(needed_gb) + float(headroom_gb)
    evicted: list[str] = []

    if free_disk_gb() >= required:
        return evicted

    repos = [r for r in _cached_repos() if getattr(r, "repo_id", None) not in keep_set]
    # Evict the least recently used snapshots first.
    repos.sort(key=lambda r: getattr(r, "last_accessed", 0))

    for repo in repos:
        if free_disk_gb() >= required:
            break
        revisions = [rev.commit_hash for rev in getattr(repo, "revisions", [])]
        if not revisions:
            continue
        try:
            from huggingface_hub import scan_cache_dir

            strategy = scan_cache_dir(cache_dir=hub_cache_dir()).delete_revisions(
                *revisions
            )
            strategy.execute()
            evicted.append(repo.repo_id)
            LOGGER.warning(
                "evicted %s from the Hub cache to free disk space", repo.repo_id
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("could not evict %s: %s", getattr(repo, "repo_id", "?"), exc)

    return evicted


def describe_runtime() -> str:
    if IS_ZEROGPU:
        backend = "ZeroGPU (Blackwell, allocated per request)"
    elif cuda_available():
        try:
            backend = _torch().cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            backend = "CUDA device"
    else:
        backend = "CPU"
    return f"{backend} · {disk_status()}"
