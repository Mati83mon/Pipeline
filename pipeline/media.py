"""Video and image I/O helpers.

The previous implementation wrapped the whole imageio write loop in a single
``try/except`` that fell through to OpenCV. That had two problems: a failure
halfway through the loop left an unclosed writer and a truncated file on disk,
and the OpenCV branch indexed ``frames[0]`` without checking that any frames
existed. Both are fixed here.
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence, Union

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("pipeline.media")

Frame = Union[Image.Image, np.ndarray]


class MediaError(RuntimeError):
    """Raised when a video cannot be written or read."""


def _as_rgb_array(frame: Frame) -> np.ndarray:
    if isinstance(frame, Image.Image):
        frame = np.asarray(frame.convert("RGB"))
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        # Diffusers can hand back float frames in [0, 1].
        array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255)
        array = array.astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise MediaError(f"unsupported frame shape {array.shape}")
    return np.ascontiguousarray(array)


def _even(value: int) -> int:
    """H.264 requires even dimensions."""
    return value - (value % 2)


def save_video(
    frames: Sequence[Frame],
    output_path: str,
    fps: int = 24,
    codec: str = "libx264",
) -> str:
    """Write frames to an MP4, preferring imageio/ffmpeg and falling back to cv2."""
    if not frames:
        raise MediaError("no frames to write")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    arrays = [_as_rgb_array(f) for f in frames]

    height, width = arrays[0].shape[:2]
    target_h, target_w = _even(height), _even(width)
    if (target_h, target_w) != (height, width):
        arrays = [a[:target_h, :target_w] for a in arrays]

    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            output_path,
            fps=int(fps),
            codec=codec,
            quality=8,
            macro_block_size=1,
        )
        try:
            for array in arrays:
                writer.append_data(array)
        finally:
            writer.close()
        return output_path
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("imageio writer failed (%s); falling back to OpenCV", exc)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)  # do not leave a truncated file behind
            except OSError:
                pass

    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path, fourcc, float(fps), (arrays[0].shape[1], arrays[0].shape[0])
        )
        if not writer.isOpened():
            raise MediaError("OpenCV could not open a video writer")
        try:
            for array in arrays:
                writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return output_path
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"could not encode video: {exc}") from exc


def extract_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Sample evenly spaced frames from a video, for vision-language input."""
    if not video_path or not os.path.exists(video_path):
        return []
    max_frames = max(1, int(max_frames))

    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(video_path)
        try:
            try:
                total = reader.count_frames()
            except Exception:  # noqa: BLE001 - some containers cannot report length
                total = 0
            if total and total > 0:
                indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
                return [Image.fromarray(reader.get_data(int(i))) for i in indices]
            # Unknown length: stream and keep a bounded, evenly-thinned window.
            collected: List[Image.Image] = []
            for position, data in enumerate(reader):
                if position % max(1, len(collected) or 1) == 0:
                    collected.append(Image.fromarray(data))
                if len(collected) >= max_frames * 4:
                    break
            if not collected:
                return []
            step = max(1, len(collected) // max_frames)
            return collected[::step][:max_frames]
        finally:
            reader.close()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("imageio reader failed (%s); falling back to OpenCV", exc)

    try:
        import cv2

        capture = cv2.VideoCapture(video_path)
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return []
            indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
            frames: List[Image.Image] = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    )
            return frames
        finally:
            capture.release()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not read %s: %s", video_path, exc)
        return []


def prune_directory(directory: str, keep: int) -> int:
    """Keep only the ``keep`` newest files in ``directory``; return count removed.

    A Space disk is ephemeral but small; unbounded generation output competes
    with the model cache for the same space.
    """
    if keep <= 0 or not os.path.isdir(directory):
        return 0
    entries = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and not name.endswith(".jsonl"):
            try:
                entries.append((os.path.getmtime(path), path))
            except OSError:
                continue
    entries.sort(reverse=True)
    removed = 0
    for _, path in entries[keep:]:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue
    return removed
