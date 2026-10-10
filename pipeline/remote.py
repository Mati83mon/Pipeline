"""Video generation delegated to another Space over its Gradio API.

Why this is a network call and not a fourth entry in ``models.py``: MiniMax-H3
cannot be loaded beside the three local models, and not for a reason that more
pins would fix.

* **Disk.** The checkpoint is 195.9 GiB in bfloat16 against a ZeroGPU Space's
  150 GB storage ceiling, which is why upstream splits it across two Spaces: a
  77.3 GB denoising half (transformer + both autoencoders) and a 66.7 GB
  Qwen3-VL conditioner. This Space already carries ~86 GB of Chroma, LTX and
  Qwen3.8. The denoising *half* alone does not fit beside them.
* **Dependencies.** The denoiser pins ``transformers==5.8.0`` and installs
  ``diffusers`` from an unmerged pull request; its own ``requirements.txt``
  records that the ``kernels`` release every FP8 attention backend wants
  "breaks transformers 5.8.0 at import". Our 27B FP8 text model *requires*
  ``kernels>=0.16`` — that is the ``finegrained-fp8 kernel unavailable`` defect
  in docs/AUDIT.md E7. No version of either package satisfies both models.
* **Allocation size.** It needs ``size="xlarge"``; upstream states ``large``
  does not fit. This Space runs ``large``, and xlarge costs 2x quota a second.

So the weights stay where they are. Everything here runs on this Space's CPU:
nothing in this module is a ``@spaces.GPU`` function, because no card is booked
on this side. The remote Space books its own — against *the requesting user's*
quota, provided the ``x-ip-token`` header is forwarded. See ``_predict``.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


class RemoteEngineError(RuntimeError):
    """A remote render failed, carrying a message meant for the user.

    A class of its own rather than a ``GenerationError`` subclass, because no
    model was loaded here and nothing on this side ran out of VRAM — reading
    one in a traceback should not send anyone looking at ``models.py``.

    ``app._fail`` treats it like the other two: no re-logging. This module
    already calls ``LOGGER.exception`` for every transport failure, and the
    rest — an unknown canvas, a duration past the ceiling, a disabled engine —
    are rejected input, which wants a clear message and no traceback at all.
    """


# --------------------------------------------------------------------------
# Constants mirrored from the remote Space, which is the authority on all of
# them. Each is re-derivable from its `app.py`; `studio_config()` checks the
# two that can drift silently.
# --------------------------------------------------------------------------

# `FPS`, and the video VAE's chunking: a decodable frame count is `17n + 5`.
FPS = 24
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5

# The canvas *label* is what goes over the wire, and the conditioner rejects a
# label it does not know, so these have to match upstream byte for byte —
# including the U+00B7 separator. The value is the (height, width) upstream
# resolves the label to; this module needs it only to estimate GPU seconds.
CANVASES: Dict[str, Tuple[int, int]] = {
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}

# Suggested sampling steps per turbo LoRA. `off` is the undistilled base model.
LORA_STEPS: Dict[str, int] = {"larry": 6, "lightx": 4, "off": 28}

# The coefficients of the remote `get_duration`. Reproduced so the UI can price
# a render *before* the click — the thing that actually matters when a 14 s clip
# at the full canvas costs minutes of quota and xlarge doubles every second.
_DUR_B, _DUR_C = 1.1745e-4, 3.8396e-9
_DECODE_BASE, _DECODE_PER_DEFAULT_CANVAS = 15, 15
_DEFAULT_CANVAS_PIXELS = 960 * 544 * 124
_PLACEMENT_ALLOWANCE, _PAD = 12, 10
_DURATION_FLOOR = 60


def snap_frames(seconds: float) -> int:
    """The frame count the remote video VAE can decode: the next ``17n + 5``.

    Identical to the remote `snap_frames`, and it has to be: the slider says
    seconds but the model renders frames, so this is the only function that
    knows what duration a request will actually come back with.
    """
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def estimate_gpu_seconds(
    canvas: str, seconds: float, steps: int, keyframes: int = 0
) -> int:
    """Seconds of GPU the remote Space will book for this request.

    A port of its `get_duration`, which is quadratic in the packed row count
    because MiniMax-H3 attends over the whole sequence every layer. Quoted to
    the user as-is; the pool's own duration factor and the xlarge 2x are
    applied on top of it by ZeroGPU, not here.
    """
    height, width = CANVASES.get(canvas, next(iter(CANVASES.values())))
    num_frames = snap_frames(seconds)
    latent_frames = (
        (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    )
    patches = (height // 32) * (width // 32)
    rows = latent_frames * patches + max(0, int(keyframes)) * patches
    denoise = max(1, int(steps)) * (_DUR_B * rows + _DUR_C * rows**2)
    decode = _DECODE_BASE + _DECODE_PER_DEFAULT_CANVAS * (
        height * width * num_frames
    ) / _DEFAULT_CANVAS_PIXELS
    return max(
        _DURATION_FLOOR, int(denoise + decode) + _PLACEMENT_ALLOWANCE + _PAD
    )


class RemoteVideoEngine:
    """A client for the MiniMax-H3 denoiser Space's ``/generate`` endpoint."""

    def __init__(self, cfg: Dict[str, Any], output_dir: str = "outputs") -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.space_id = str(cfg.get("space_id", "")).strip()
        self.api_name = str(cfg.get("api_name", "/generate"))
        self._client: Any = None
        self._studio: Optional[Dict[str, Any]] = None
        self._studio_checked = False
        # One client, built once and reused. The per-caller header rides on the
        # `predict` call instead, so a new client per request would only buy a
        # redundant round trip for the Space's config.
        self._lock = threading.Lock()

    # -- metadata ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False)) and bool(self.space_id)

    @property
    def label(self) -> str:
        return str(self.cfg.get("label") or self.space_id)

    def canvases(self) -> List[str]:
        """Dropdown choices. Local only — this is on the UI build path.

        Deliberately does *not* ask the remote Space, even though the remote
        table is the authority: `build_ui()` calls this, and a Space that is
        mid-load answers slowly or not at all. Letting that reach the build
        path would add seconds to every start of *this* Space, including the
        starts where nobody opens this tab. `check_remote_canvases()` does the
        asking, from `status()`, where a person is waiting for an answer.
        """
        configured = self.cfg.get("canvases")
        if isinstance(configured, list) and configured:
            return [str(item) for item in configured]
        return list(CANVASES)

    def default_canvas(self) -> str:
        choices = self.canvases()
        preferred = str(self.cfg.get("default_canvas", ""))
        return preferred if preferred in choices else choices[0]

    def loras(self) -> Dict[str, int]:
        configured = self.cfg.get("loras")
        if isinstance(configured, dict) and configured:
            return {str(k): int(v) for k, v in configured.items()}
        return dict(LORA_STEPS)

    def duration_bounds(self) -> Tuple[float, float]:
        """The slider's range.

        The upper bound is the checkpoint's, not a UI preference: 15 s is 360
        frames, which snaps up to 362 (15.083 s) and the pipeline refuses it.
        14 s snaps to 345 frames — 14.375 s of output — and is the longest
        request that lands.
        """
        return (
            float(self.cfg.get("min_duration", 2)),
            float(self.cfg.get("max_duration", 14)),
        )

    def estimate(
        self, canvas: str, duration: float, steps: int, keyframes: int = 0
    ) -> int:
        """`estimate_gpu_seconds` against this engine's own canvas default."""
        return estimate_gpu_seconds(
            canvas or self.default_canvas(), duration, steps, keyframes
        )

    def studio_config(self) -> Optional[Dict[str, Any]]:
        """The remote ``/studio-config``, fetched once, ``None`` on any failure.

        Advisory, and never on the UI build path. A Space that is still loading
        its 77.3 GB answers slowly or not at all, and this must not be able to
        delay or break anything that merely renders the tab.
        """
        if self._studio_checked:
            return self._studio
        self._studio_checked = True
        if not self.space_id:
            return None
        try:
            import httpx

            url = f"https://{self._subdomain()}.hf.space/studio-config"
            response = httpx.get(
                url, timeout=float(self.cfg.get("probe_timeout_seconds", 8))
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                self._studio = payload
        except Exception as exc:  # noqa: BLE001
            LOGGER.info(
                "remote studio-config unavailable for %s (%s: %s); using config.json",
                self.space_id,
                type(exc).__name__,
                exc,
            )
        return self._studio

    def _subdomain(self) -> str:
        """``owner/name`` as Spaces spells it in a hostname."""
        return (
            self.space_id.replace("/", "-").replace("_", "-").replace(".", "-").lower()
        )

    def check_remote_canvases(self) -> Optional[str]:
        """Compare our canvas table against the remote's. ``None`` when equal.

        Worth a check because the failure it catches is otherwise unreadable:
        the label is matched by string on the far side, so one renamed entry
        upstream turns every render into an error raised by a conditioner the
        user has never heard of.
        """
        remote = (self.studio_config() or {}).get("canvases")
        if not isinstance(remote, list) or not remote:
            return None
        ours, theirs = set(self.canvases()), {str(item) for item in remote}
        if ours == theirs:
            return None
        missing = sorted(theirs - ours)
        stale = sorted(ours - theirs)
        parts = []
        if stale:
            parts.append(f"no longer offered: {', '.join(stale)}")
        if missing:
            parts.append(f"new upstream: {', '.join(missing)}")
        return (
            f"{self.space_id} changed its canvas table — "
            + "; ".join(parts)
            + ". Update `video.remote_engine.canvases` in config.json."
        )

    def status(self) -> str:
        if not self.enabled:
            return "MiniMax-H3: disabled in config.json"
        drift = self.check_remote_canvases()
        state = "reachable" if self._studio else "not reached"
        line = (
            f"MiniMax-H3 via {self.space_id} · {len(self.canvases())} canvases · "
            f"{state}"
        )
        return f"{line}\n{drift}" if drift else line

    # -- generation --------------------------------------------------------

    def generate(
        self,
        *,
        prompt: str,
        image: Optional[str] = None,
        last_image: Optional[str] = None,
        canvas: Optional[str] = None,
        duration: float = 5.0,
        steps: int = 6,
        seed: int = 42,
        upsample: bool = False,
        lora: str = "larry",
        ip_token: Optional[str] = None,
    ) -> Tuple[str, str, str, float]:
        """One remote render. Returns ``(path, report, refined_prompt, seconds)``.

        ``path`` is in this Space's own output directory, so the History tab and
        the output pruner treat a remote render exactly like a local one.
        """
        if not self.enabled:
            raise RemoteEngineError(
                "The MiniMax-H3 engine is disabled. Set "
                "`video.remote_engine.enabled` and `space_id` in config.json."
            )
        choices = self.canvases()
        canvas = canvas or self.default_canvas()
        if canvas not in choices:
            raise RemoteEngineError(
                f"`{canvas}` is not a canvas {self.space_id} accepts. "
                f"Pick one of: {', '.join(choices)}."
            )
        low, high = self.duration_bounds()
        if not low <= float(duration) <= high:
            raise RemoteEngineError(
                f"Duration must be between {low:g} and {high:g} s. "
                f"{high:g} s is the checkpoint's ceiling, not a UI limit: one "
                f"second more snaps past the longest frame count its video VAE "
                f"can decode and the request is refused."
            )

        started = time.time()
        video, report, refined = self._predict(
            prompt=prompt,
            image=image,
            last_image=last_image,
            canvas=canvas,
            duration=float(duration),
            steps=int(steps),
            seed=int(seed),
            upsample=bool(upsample),
            lora=str(lora),
            ip_token=ip_token,
        )
        elapsed = time.time() - started
        return self._adopt(video, seed), report, refined, elapsed

    def _predict(
        self,
        *,
        prompt: str,
        image: Optional[str],
        last_image: Optional[str],
        canvas: str,
        duration: float,
        steps: int,
        seed: int,
        upsample: bool,
        lora: str,
        ip_token: Optional[str],
    ) -> Tuple[str, str, str]:
        """The call itself, with the caller's ZeroGPU identity attached.

        ``x-ip-token`` is the whole reason this takes a `gr.Request`. ZeroGPU
        bills a booking to whatever that header identifies, so forwarding it
        charges the render to the person who clicked. Drop it and both
        bookings — the denoiser's and the conditioner's — land on *this*
        Space's pod IP and its small shared quota, where an unattributed caller
        may hold at most 120 credits and an xlarge second costs two of them. A
        14 s clip is several hundred seconds of xlarge: it does not fit in that
        ceiling, so the failure mode is not a bigger bill, it is every render
        queueing forever for an allocation it can never be granted.

        The remote Space solves the identical problem for its own conditioner
        call the same way, and its comment says why the obvious alternative
        does not work: ``LocalContext``-based forwarding "is not reliable in
        Server mode".
        """
        try:
            from gradio_client import handle_file
        except ImportError as exc:  # pragma: no cover - gradio ships it
            raise RemoteEngineError(
                "`gradio_client` is missing. It arrives with gradio itself "
                "(gradio 6.26.0 pins gradio-client 2.6.1), so a missing one "
                "means the install is broken rather than under-specified."
            ) from exc

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "image_path": handle_file(image) if image else None,
            "last_image_path": handle_file(last_image) if last_image else None,
            "canvas": canvas,
            "duration": duration,
            "steps": steps,
            "seed": seed,
            "upsample": upsample,
            "lora": lora,
        }
        headers = {"x-ip-token": ip_token} if ip_token else None
        try:
            result = self._connect().predict(
                api_name=self.api_name, headers=headers, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("remote generation failed on %s", self.space_id)
            raise RemoteEngineError(self._explain(exc, bool(ip_token))) from exc

        return self._unpack(result)

    def _connect(self):
        """One cached client.

        `HF_TOKEN` is **not** sent by default, and that is a security decision
        rather than an omission. The default `space_id` is a Space this project
        does not own; our token is a read credential for a gated model, and
        attaching it to every request would hand it to whoever runs that Space
        — for nothing, because the default target is public and needs no
        credential. A *private* duplicate does need one, so the config can opt
        in per Space with `send_hf_token`. Only ever set it on a Space you own.
        """
        with self._lock:
            if self._client is not None:
                return self._client
            from gradio_client import Client

            token = None
            if bool(self.cfg.get("send_hf_token", False)):
                token = os.environ.get("HF_TOKEN") or None
                if token:
                    LOGGER.warning(
                        "sending HF_TOKEN to %s because video.remote_engine."
                        "send_hf_token is set — only do this for a Space you own",
                        self.space_id,
                    )
            self._client = Client(
                self.space_id,
                token=token,
                verbose=False,
                analytics_enabled=False,
                httpx_kwargs={"timeout": float(self.cfg.get("timeout_seconds", 1800))},
            )
            return self._client

    @staticmethod
    def _unpack(result: Any) -> Tuple[str, str, str]:
        """``(video, report, refined_prompt)``, tolerating a shorter tuple.

        The endpoint is typed `tuple[FileData, str, str]` and `gradio_client`
        resolves the FileData to a local path. `refined` is empty unless the
        prompt was upsampled, and a future revision adding a fourth element
        should not break this tab.
        """
        if isinstance(result, (list, tuple)):
            parts = list(result) + ["", "", ""]
        else:
            parts = [result, "", ""]
        video, report, refined = parts[0], parts[1], parts[2]
        if isinstance(video, dict):
            video = video.get("path") or video.get("url") or ""
        if not video:
            raise RemoteEngineError(
                "The remote Space returned no video. Its logs will say why."
            )
        return str(video), str(report or ""), str(refined or "")

    def _adopt(self, remote_path: str, seed: int) -> str:
        """Move the downloaded file into this Space's output directory.

        `gradio_client` drops it in a temp directory it manages, which neither
        the History tab nor `prune_directory` knows about — so a remote render
        would be the one output that never appears in history and never gets
        pruned.
        """
        from datetime import datetime

        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(self.output_dir, f"video_h3_{stamp}_{int(seed)}.mp4")
        try:
            shutil.move(remote_path, target)
        except OSError:
            # Across filesystems, or a path the client still holds open.
            shutil.copyfile(remote_path, target)
        return target

    def _explain(self, exc: Exception, had_token: bool) -> str:
        """Turn a transport failure into something actionable.

        Each branch is a failure this integration can actually produce, and the
        generic message for any of them ("connection error") points at the
        wrong layer every time.
        """
        text = f"{type(exc).__name__}: {exc}"
        lowered = text.lower()
        if "not found" in lowered or "404" in lowered:
            return (
                f"{self.space_id} did not answer at `{self.api_name}`. Check the "
                f"Space still exists and is public, or point "
                f"`video.remote_engine.space_id` at your own duplicate. ({text})"
            )
        if "still loading" in lowered or "denoiser" in lowered:
            return (
                f"{self.space_id} is still loading — it downloads 77.3 GB on a "
                f"cold start, about 93 s warm and several minutes cold. Try "
                f"again shortly. ({text})"
            )
        if "quota" in lowered or ("gpu" in lowered and "credits" in lowered):
            quota = (
                "your ZeroGPU quota is spent"
                if had_token
                else "the booking was not attributed to you, so it fell back to "
                "this Space's small shared quota"
            )
            return f"The remote render could not book a GPU: {quota}. ({text})"
        if "timeout" in lowered or "timed out" in lowered:
            return (
                f"{self.space_id} did not finish within "
                f"{self.cfg.get('timeout_seconds', 1800)} s. A 14 s clip at the "
                f"full canvas is several hundred seconds of GPU plus the queue; "
                f"raise `video.remote_engine.timeout_seconds` or ask for a "
                f"shorter clip. ({text})"
            )
        return f"{self.space_id} failed this render. {text}"
