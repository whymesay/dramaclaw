"""MiniMax H3 image-reference video workflow on AutoDL."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .client import AutoDLWorkflowClient
from .workflows import MINIMAX_H3_IMAGE_REFERENCE

AUTODL_MINIMAX_H3_BACKEND = "autodl_minimax-h3"


def minimax_h3_resolution(resolution: str | None, aspect_ratio: str | None) -> str:
    value = str(resolution or "768p").strip().lower()
    if "480" in value:
        quality = "480p"
    elif "1080" in value:
        quality = "1080p"
    elif "768" in value:
        quality = "768p"
    else:
        raise ValueError("MiniMax H3 resolution must be 480p, 768p, or 1080p")
    ratio = str(aspect_ratio or "9:16").strip()
    if ratio not in {"9:16", "16:9"}:
        raise ValueError("MiniMax H3 aspect_ratio must be 9:16 or 16:9")
    return f"{quality}{'竖' if ratio == '9:16' else '横'}"


class AutoDLMinimaxH3ImageReferenceGenerator:
    def __init__(
        self,
        *,
        resolution: str = "768p",
        client: AutoDLWorkflowClient | None = None,
        **_: Any,
    ) -> None:
        self.resolution = resolution
        self.client = client or AutoDLWorkflowClient()

    @staticmethod
    def _image_paths(image_path: str | None, references: list[Any] | None) -> list[str]:
        values = [str(image_path or "").strip()]
        values.extend(
            str(getattr(ref, "path", "") or "").strip()
            for ref in references or []
            if str(getattr(ref, "type", "image") or "image").lower() == "image"
        )
        return list(dict.fromkeys(value for value in values if value))

    @staticmethod
    def _to_data_url(value: str) -> str:
        if value.startswith("data:image/") and ";base64," in value:
            return value
        path = Path(value)
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError(f"unsupported MiniMax H3 reference image: {value}")
        if not path.is_file():
            raise ValueError(f"MiniMax H3 reference image not found: {value}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    async def generate(
        self,
        image_path: str | None,
        prompt: str,
        output_path: str,
        aspect_ratio: str = "9:16",
        duration: float = 5.0,
        poll_interval: float = 5.0,
        max_polls: int = 360,
        references: list[Any] | None = None,
        **_: Any,
    ):
        from novelvideo.generators.video_generator import VideoGenResult, VideoGenStatus

        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            return VideoGenResult(
                status=VideoGenStatus.FAILED, error="prompt is required"
            )
        images = self._image_paths(image_path, references)
        if not 1 <= len(images) <= 9:
            return VideoGenResult(
                status=VideoGenStatus.FAILED,
                error="MiniMax H3 image reference mode requires 1 to 9 images",
            )
        seconds = int(duration)
        if not 1 <= seconds <= 10 or seconds != duration:
            return VideoGenResult(
                status=VideoGenStatus.FAILED,
                error="duration must be an integer from 1 to 10",
            )
        try:
            payload: dict[str, Any] = {
                "prompt": clean_prompt,
                "duration": seconds,
                "resolution": minimax_h3_resolution(self.resolution, aspect_ratio),
            }
            payload.update(
                {
                    f"ref_image_{index}": self._to_data_url(value)
                    for index, value in enumerate(images)
                }
            )
            task_id = await self.client.submit(MINIMAX_H3_IMAGE_REFERENCE, payload)
            result = await self.client.wait_for_result(
                MINIMAX_H3_IMAGE_REFERENCE,
                task_id,
                poll_interval=poll_interval,
                max_polls=max_polls,
            )
            await self.client.download(result.output_url, output_path)
            return VideoGenResult(
                status=VideoGenStatus.DONE,
                video_url=result.output_url,
                video_path=output_path,
                task_id=task_id,
                provider_task_id=task_id,
                duration_seconds=float(seconds),
            )
        except Exception as exc:
            return VideoGenResult(status=VideoGenStatus.FAILED, error=str(exc))
