"""Model loading and generation.

Design notes
------------
*One model resident at a time.* The three model families total roughly 85 GB
on disk and far more than a single ZeroGPU allocation in VRAM, so the manager
keeps exactly one loaded and evicts the previous one — including from the Hub
disk cache when space runs short.

*CPU-resident, GPU-transient.* Weights are materialised in host RAM outside any
GPU task and moved to the accelerator inside ``@spaces.GPU`` for the duration of
a request. That is the only shape that works unchanged on ZeroGPU, on a
dedicated GPU Space, in Docker and on a CPU box.

*Loader arguments are probed, not assumed.* ``diffusers`` still takes
``torch_dtype`` while ``transformers`` v5 renamed it to ``dtype``; both spellings
are attempted so the app survives either side moving.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .config import AppConfig
from .history import GenerationLogger
from .media import extract_frames, prune_directory, save_video
from .runtime import (
    ensure_disk_budget,
    free_vram,
    hf_token,
    preferred_device,
    resolve_dtype,
)

LOGGER = logging.getLogger("pipeline.models")

MODULES = ("image", "video", "llm")


class ModelLoadError(RuntimeError):
    """Raised when neither the primary nor the fallback model can be loaded."""


class GenerationError(RuntimeError):
    """Raised when a loaded model fails to produce output."""


@dataclass
class LoadedModel:
    module: str
    model_id: str
    handle: Any
    is_fallback: bool
    extra: Dict[str, Any]


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _call_with_dtype(loader: Callable[..., Any], *args: Any, dtype: Any, **kwargs: Any):
    """Call a ``from_pretrained`` that may spell the dtype argument either way."""
    try:
        return loader(*args, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return loader(*args, torch_dtype=dtype, **kwargs)


def _diffusers_class(name: str):
    import diffusers

    try:
        return getattr(diffusers, name)
    except AttributeError as exc:
        raise ModelLoadError(
            f"diffusers {diffusers.__version__} has no pipeline class '{name}'. "
            "Update the diffusers pin in requirements.txt or fix "
            "config.json -> pipeline_class."
        ) from exc


def _enable_memory_savers(pipe: Any) -> None:
    """Turn on tiling/slicing where the pipeline supports it."""
    for target, method in (
        ("vae", "enable_tiling"),
        ("vae", "enable_slicing"),
    ):
        component = getattr(pipe, target, None)
        fn = getattr(component, method, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("%s.%s failed: %s", target, method, exc)


def align(value: int, multiple: int, minimum: int) -> int:
    """Round ``value`` down to a multiple, never below ``minimum``."""
    aligned = (int(value) // multiple) * multiple
    return max(minimum, aligned)


def align_frames(num_frames: int, minimum: int = 9) -> int:
    """LTX needs ``8n + 1`` frames."""
    n = max(1, (int(num_frames) - 1) // 8)
    return max(minimum, n * 8 + 1)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ModelManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.system = config.system
        self.output_dir = self.system.get("output_dir", "outputs")
        self.logger = GenerationLogger(
            self.system.get("log_file", "outputs/generation_log.jsonl"),
            max_entries=int(self.system.get("log_max_entries", 2000)),
        )
        self._loaded: Optional[LoadedModel] = None
        self._lock = threading.RLock()
        self._notices: List[str] = []
        os.makedirs(self.output_dir, exist_ok=True)

    # -- introspection -----------------------------------------------------

    @property
    def loaded_module(self) -> Optional[str]:
        return self._loaded.module if self._loaded else None

    def status(self) -> str:
        if not self._loaded:
            return "No model loaded"
        suffix = " (fallback)" if self._loaded.is_fallback else ""
        return f"{self._loaded.module}: {self._loaded.model_id}{suffix}"

    def take_notices(self) -> List[str]:
        with self._lock:
            notices, self._notices = self._notices, []
        return notices

    def _notice(self, message: str) -> None:
        LOGGER.info(message)
        self._notices.append(message)

    # -- lifecycle ---------------------------------------------------------

    def unload(self) -> None:
        with self._lock:
            if not self._loaded:
                return
            LOGGER.info("unloading %s", self._loaded.model_id)
            handle = self._loaded.handle
            for candidate in (handle, *(self._loaded.extra or {}).values()):
                to = getattr(candidate, "to", None)
                if callable(to):
                    try:
                        to("cpu")
                    except Exception:  # noqa: BLE001
                        pass
            self._loaded = None
            free_vram()

    def ensure_loaded(self, module: str) -> LoadedModel:
        """Load ``module`` if it is not already resident. Never call inside a GPU task."""
        if module not in MODULES:
            raise ValueError(f"unknown module '{module}'")
        with self._lock:
            if self._loaded and self._loaded.module == module:
                return self._loaded

            if not self.config.is_enabled(module):
                raise ModelLoadError(
                    f"module '{module}' is disabled in config.json "
                    f"(set image/video/llm -> enabled: true to turn it back on)"
                )

            if self._loaded and self.system.get("unload_previous_model", True):
                self.unload()

            cfg = self.config.section(module)
            self._make_room(cfg)

            primary = cfg["model_id"]
            fallback = cfg.get("fallback_model_id")
            started = time.time()

            try:
                handle, extra = self._load(module, primary, cfg)
                self._loaded = LoadedModel(module, primary, handle, False, extra)
            except Exception as primary_exc:  # noqa: BLE001
                # Log the traceback, not just the message. A silent fallback
                # whose only trace is the exception class name is unusable:
                # it cannot distinguish a gated 401 from an unsupported
                # architecture, an OOM or a dtype conflict.
                LOGGER.exception("loading %s failed", primary)
                if not fallback:
                    raise ModelLoadError(
                        self._explain_load_failure(primary, primary_exc)
                    ) from primary_exc
                self._notice(
                    f"{primary} could not be loaded — {self._short_reason(primary_exc)}. "
                    f"Falling back to {fallback}. Full traceback is in the Space logs."
                )
                try:
                    handle, extra = self._load(module, fallback, cfg)
                except Exception as fallback_exc:  # noqa: BLE001
                    raise ModelLoadError(
                        f"{self._explain_load_failure(primary, primary_exc)}\n"
                        f"Fallback {fallback} also failed: {fallback_exc}"
                    ) from fallback_exc
                self._loaded = LoadedModel(module, fallback, handle, True, extra)

            LOGGER.info(
                "%s ready in %.1fs (%s)",
                module,
                time.time() - started,
                self._loaded.model_id,
            )
            return self._loaded

    def _make_room(self, cfg: Dict[str, Any]) -> None:
        if not self.system.get("disk_guard_enabled", True):
            return
        needed = float(cfg.get("approx_download_gb", 20))
        headroom = float(self.system.get("disk_headroom_gb", 10))
        keep = [cfg["model_id"]]
        if cfg.get("fallback_model_id"):
            keep.append(cfg["fallback_model_id"])
        evicted = ensure_disk_budget(needed, keep=keep, headroom_gb=headroom)
        if evicted:
            self._notice(
                "Freed disk space by evicting cached models: "
                + ", ".join(evicted)
                + ". They will re-download when next used."
            )

    @staticmethod
    def _short_reason(exc: Exception, limit: int = 240) -> str:
        """One-line cause for the UI: exception type plus its first line.

        The UI previously showed only the class name, which for a chain of
        loader attempts is always ``ModelLoadError`` — true and useless.
        """
        text = " ".join(str(exc).split())
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__

    @staticmethod
    def _explain_load_failure(model_id: str, exc: Exception) -> str:
        text = str(exc)
        lowered = text.lower()
        if "401" in text or "gated" in lowered or "authorized" in lowered:
            return (
                f"{model_id} is gated or private. Accept the licence on its model "
                f"page and add an HF_TOKEN secret to this Space "
                f"(Settings -> Variables and secrets). Original error: {text}"
            )
        if "404" in text or "not found" in lowered:
            return f"{model_id} does not exist on the Hub. Original error: {text}"
        if "no space left" in lowered or "disk" in lowered:
            return (
                f"Ran out of disk while fetching {model_id}. Lower "
                f"system.disk_headroom_gb or disable a module in config.json. "
                f"Original error: {text}"
            )
        if "out of memory" in lowered:
            return (
                f"Out of memory loading {model_id}. Use a smaller model or a "
                f"larger GPU size. Original error: {text}"
            )
        return f"Could not load {model_id}: {text}"

    # -- concrete loaders --------------------------------------------------

    def _load(
        self, module: str, model_id: str, cfg: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        if module == "image":
            return self._load_image(model_id, cfg)
        if module == "video":
            return self._load_video(model_id, cfg)
        return self._load_llm(model_id, cfg)

    def _load_image(self, model_id: str, cfg: Dict[str, Any]):
        pipeline_cls = _diffusers_class(cfg.get("pipeline_class", "ChromaPipeline"))
        pipe = _call_with_dtype(
            pipeline_cls.from_pretrained,
            model_id,
            dtype=resolve_dtype(cfg.get("dtype", "bfloat16")),
            token=hf_token(),
        )
        _enable_memory_savers(pipe)
        pipe.set_progress_bar_config(disable=True)
        return pipe, {}

    def _load_video(self, model_id: str, cfg: Dict[str, Any]):
        pipeline_cls = _diffusers_class(cfg.get("pipeline_class", "LTXPipeline"))
        pipe = _call_with_dtype(
            pipeline_cls.from_pretrained,
            model_id,
            dtype=resolve_dtype(cfg.get("dtype", "bfloat16")),
            token=hf_token(),
        )
        _enable_memory_savers(pipe)
        pipe.set_progress_bar_config(disable=True)

        # Image-to-video reuses the same weights through a sibling pipeline, so
        # switching modes costs no extra VRAM and no extra download.
        extra: Dict[str, Any] = {}
        i2v_name = cfg.get("image_pipeline_class")
        if i2v_name:
            try:
                i2v_cls = _diffusers_class(i2v_name)
                extra["i2v"] = i2v_cls(**pipe.components)
                extra["i2v"].set_progress_bar_config(disable=True)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("image-to-video pipeline unavailable: %s", exc)
        return pipe, extra

    @staticmethod
    def _declared_quantization(
        model_id: str, trust: bool, token: Optional[str]
    ) -> Optional[str]:
        """Return the quantisation method a repo declares in its own config.

        Some checkpoints ship pre-quantised weights and describe the scheme in
        ``config.json`` — the default text model here is block-FP8 with
        ``weight_block_size: [128, 128]``, built for the vLLM kernel path.
        Handing such a repo an explicit ``dtype`` overrides the policy the
        weights were produced under, so the caller needs to know.
        """
        try:
            import transformers

            config = transformers.AutoConfig.from_pretrained(
                model_id, trust_remote_code=trust, token=token
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("could not read the config of %s: %s", model_id, exc)
            return None

        quant = getattr(config, "quantization_config", None)
        if not quant:
            return None
        if isinstance(quant, dict):
            return str(quant.get("quant_method") or "quantized")
        return str(getattr(quant, "quant_method", "quantized"))

    def _load_llm(self, model_id: str, cfg: Dict[str, Any]):
        import transformers

        trust = bool(cfg.get("trust_remote_code", False))
        token = hf_token()

        # A repo that carries its own quantisation must be loaded on the dtype
        # policy its config declares. Forcing one on top either errors outright
        # or silently dequantises tens of GB of weights. `dtype: "force"` in
        # config.json opts back into the explicit value.
        dtype_name = str(cfg.get("dtype", "bfloat16"))
        declared_quant = self._declared_quantization(model_id, trust, token)
        if declared_quant and not dtype_name.startswith("force"):
            dtype = None
            LOGGER.info(
                "%s declares %s quantisation; leaving dtype to the checkpoint",
                model_id,
                declared_quant,
            )
        else:
            dtype = resolve_dtype(dtype_name.removeprefix("force:") or "bfloat16")

        processor = None
        tokenizer = None
        if cfg.get("multimodal", True):
            try:
                processor = transformers.AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=trust, token=token
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("no processor for %s (%s); text-only mode", model_id, exc)
        if processor is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust, token=token, padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust,
            "token": token,
            "low_cpu_mem_usage": True,
        }
        if declared_quant:
            # Stacking bitsandbytes on top of weights that are already
            # quantised is not a supported combination.
            if cfg.get("load_in_4bit") or cfg.get("load_in_8bit"):
                self._notice(
                    f"Ignoring load_in_4bit/8bit for {model_id}: the checkpoint "
                    f"is already {declared_quant}-quantised."
                )
        else:
            quant = self._quantization_config(cfg, dtype)
            if quant is not None:
                load_kwargs["quantization_config"] = quant

        # Try the most specific auto class first and fall back. The Hub lists
        # this model under AutoModelForMultimodalLM while its config declares
        # Qwen3_5ForConditionalGeneration, so both spellings are attempted
        # before the text-only class.
        auto_classes = []
        if processor is not None:
            for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
                auto_cls = getattr(transformers, name, None)
                if auto_cls is not None:
                    auto_classes.append(auto_cls)
        auto_classes.append(transformers.AutoModelForCausalLM)

        last_exc: Optional[Exception] = None
        model = None
        for auto_cls in auto_classes:
            try:
                if dtype is None:
                    model = auto_cls.from_pretrained(model_id, **load_kwargs)
                else:
                    model = _call_with_dtype(
                        auto_cls.from_pretrained, model_id, dtype=dtype, **load_kwargs
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # Full traceback, not just str(exc): when every auto class
                # fails, the reason is the only thing that distinguishes an
                # unsupported architecture from a 401, an OOM or a bad dtype.
                LOGGER.exception("%s could not load %s", auto_cls.__name__, model_id)
        if model is None:
            raise ModelLoadError(
                f"no transformers auto class could load {model_id} "
                f"(tried {', '.join(c.__name__ for c in auto_classes)}): {last_exc}"
            )

        model.eval()
        if processor is None and tokenizer is None:  # pragma: no cover - defensive
            raise ModelLoadError(f"no tokenizer or processor available for {model_id}")
        return model, {"processor": processor, "tokenizer": tokenizer}

    @staticmethod
    def _quantization_config(cfg: Dict[str, Any], dtype: Any):
        if not (cfg.get("load_in_4bit") or cfg.get("load_in_8bit")):
            return None
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("bitsandbytes quantisation unavailable: %s", exc)
            return None
        if cfg.get("load_in_4bit"):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    # -- device movement (called inside a GPU task) ------------------------

    def _to_device(self, loaded: LoadedModel, device: str) -> Any:
        handle = loaded.handle
        try:
            handle = handle.to(device)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                f"could not move {loaded.model_id} to {device}: {exc}"
            ) from exc
        for value in (loaded.extra or {}).values():
            mover = getattr(value, "to", None)
            if callable(mover):
                try:
                    mover(device)
                except Exception:  # noqa: BLE001
                    pass
        return handle

    def _release_device(self, loaded: LoadedModel) -> None:
        if not self.system.get("clear_cache_after_gen", True):
            return
        try:
            loaded.handle.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        free_vram()

    # -- generation --------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: int,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Any, str, float]:
        import torch

        loaded = self.ensure_loaded("image")
        cfg = self.config.image
        device = preferred_device()

        width = align(width, cfg.get("size_step", 64), cfg.get("min_size", 512))
        height = align(height, cfg.get("size_step", 64), cfg.get("min_size", 512))

        pipe = self._to_device(loaded, device)
        generator = torch.Generator(device=device).manual_seed(int(seed))

        started = time.time()
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=width,
                height=height,
                generator=generator,
                callback_on_step_end=progress_callback,
            )
        except Exception as exc:  # noqa: BLE001
            self._release_device(loaded)
            raise GenerationError(self._explain_generation_failure(exc)) from exc
        duration = time.time() - started

        image = result.images[0]
        out_path = self._output_path("image", seed, "png")
        image.save(out_path)
        self._finish(
            loaded,
            "image",
            prompt,
            {
                "negative_prompt": negative_prompt,
                "steps": int(steps),
                "guidance": float(guidance),
                "width": width,
                "height": height,
                "seed": int(seed),
            },
            duration,
            out_path,
        )
        return image, out_path, duration

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        image: Optional[Image.Image],
        num_frames: int,
        fps: int,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: int,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[str, float]:
        import torch

        loaded = self.ensure_loaded("video")
        cfg = self.config.video
        device = preferred_device()

        width = align(width, cfg.get("size_step", 32), cfg.get("min_size", 320))
        height = align(height, cfg.get("size_step", 32), cfg.get("min_size", 320))
        frames_aligned = align_frames(num_frames, cfg.get("min_frames", 9))

        pipe = self._to_device(loaded, device)
        if image is not None:
            i2v = (loaded.extra or {}).get("i2v")
            if i2v is None:
                raise GenerationError(
                    "This video model has no image-to-video pipeline; clear the "
                    "input image to run text-to-video."
                )
            pipe = i2v

        generator = torch.Generator(device=device).manual_seed(int(seed))
        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or cfg.get("default_negative_prompt"),
            "width": width,
            "height": height,
            "num_frames": frames_aligned,
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "generator": generator,
            "callback_on_step_end": progress_callback,
        }
        if image is not None:
            call_kwargs["image"] = image.convert("RGB").resize((width, height))

        started = time.time()
        try:
            result = pipe(**call_kwargs)
        except Exception as exc:  # noqa: BLE001
            self._release_device(loaded)
            raise GenerationError(self._explain_generation_failure(exc)) from exc
        duration = time.time() - started

        frames = result.frames[0]
        out_path = self._output_path("video", seed, "mp4")
        save_video(frames, out_path, fps=int(fps))
        self._finish(
            loaded,
            "video",
            prompt,
            {
                "negative_prompt": call_kwargs["negative_prompt"],
                "num_frames": frames_aligned,
                "fps": int(fps),
                "steps": int(steps),
                "guidance": float(guidance),
                "width": width,
                "height": height,
                "seed": int(seed),
                "image_to_video": image is not None,
            },
            duration,
            out_path,
        )
        return out_path, duration

    def generate_text(
        self,
        prompt: str,
        image: Optional[Image.Image],
        video: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> Tuple[str, str, float]:
        import torch

        loaded = self.ensure_loaded("llm")
        cfg = self.config.llm
        device = preferred_device()
        processor = (loaded.extra or {}).get("processor")
        tokenizer = (loaded.extra or {}).get("tokenizer")

        media: List[Image.Image] = []
        if image is not None:
            media.append(image.convert("RGB"))
        if video:
            media.extend(extract_frames(video, int(cfg.get("max_video_frames", 8))))

        if media and processor is None:
            self._notice(
                f"{loaded.model_id} is a text-only model; the attached image/video "
                "was ignored."
            )
            media = []

        model = self._to_device(loaded, device)
        torch.manual_seed(int(seed))

        inputs = self._build_llm_inputs(cfg, processor, tokenizer, prompt, media)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = float(temperature) > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            # Passing temperature/top_p with do_sample=False is a hard error in
            # recent transformers releases, not just a warning.
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        pad_id = self._pad_token_id(processor, tokenizer)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id

        started = time.time()
        try:
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)
        except Exception as exc:  # noqa: BLE001
            self._release_device(loaded)
            raise GenerationError(self._explain_generation_failure(exc)) from exc
        duration = time.time() - started

        prompt_len = inputs["input_ids"].shape[1]
        decoder = processor if processor is not None else tokenizer
        response = decoder.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

        out_path = self._output_path("text", seed, "txt")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(f"MODEL: {loaded.model_id}\n\nPROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n")

        self._finish(
            loaded,
            "text",
            prompt,
            {
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "seed": int(seed),
                "images": 1 if image is not None else 0,
                "video_frames": max(0, len(media) - (1 if image is not None else 0)),
            },
            duration,
            out_path,
        )
        return response, out_path, duration

    # -- LLM input assembly ------------------------------------------------

    def _build_llm_inputs(
        self,
        cfg: Dict[str, Any],
        processor: Any,
        tokenizer: Any,
        prompt: str,
        media: List[Image.Image],
    ) -> Dict[str, Any]:
        system_prompt = cfg.get("system_prompt") or ""

        if processor is not None:
            content: List[Dict[str, Any]] = [
                {"type": "image", "image": img} for img in media
            ]
            content.append({"type": "text", "text": prompt})
            messages = []
            if system_prompt:
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                )
            messages.append({"role": "user", "content": content})
            try:
                return processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "processor chat template failed (%s); using text-only path", exc
                )
                tokenizer = getattr(processor, "tokenizer", None) or tokenizer

        if tokenizer is None:
            raise GenerationError("no tokenizer available for this model")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        return dict(tokenizer(text, return_tensors="pt"))

    @staticmethod
    def _pad_token_id(processor: Any, tokenizer: Any) -> Optional[int]:
        source = tokenizer or getattr(processor, "tokenizer", None)
        if source is None:
            return None
        return getattr(source, "pad_token_id", None) or getattr(
            source, "eos_token_id", None
        )

    # -- shared tail -------------------------------------------------------

    @staticmethod
    def _explain_generation_failure(exc: Exception) -> str:
        text = str(exc)
        lowered = text.lower()
        if "out of memory" in lowered or "cuda oom" in lowered:
            return (
                "Out of GPU memory. Lower the resolution, frame count or step "
                f"count and try again. Original error: {text}"
            )
        if "gpu task aborted" in lowered or "timeout" in lowered:
            return (
                "The GPU allocation expired before the run finished. Reduce steps "
                f"or frames. Original error: {text}"
            )
        return text

    def _output_path(self, kind: str, seed: int, extension: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{kind}_{stamp}_{int(seed)}.{extension}"
        return os.path.join(self.output_dir, name)

    def _finish(
        self,
        loaded: LoadedModel,
        kind: str,
        prompt: str,
        params: Dict[str, Any],
        duration: float,
        out_path: str,
    ) -> None:
        self.logger.log(
            gen_type=kind,
            prompt=prompt,
            params=params,
            duration=duration,
            output_path=out_path,
            model_id=loaded.model_id,
            fallback=loaded.is_fallback,
        )
        prune_directory(self.output_dir, int(self.system.get("keep_outputs", 40)))
        self._release_device(loaded)
