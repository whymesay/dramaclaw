"""Shared authenticated client for asynchronous AutoDL workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from .workflows import AutoDLWorkflowSpec


@dataclass(frozen=True)
class AutoDLTaskResult:
    task_id: str
    status: str
    output_url: str = ""
    request_id: str = ""
    client_id: str = ""
    message: str = ""


class AutoDLWorkflowClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        from novelvideo.model_gateway_settings import get_effective_autodl_config

        effective = get_effective_autodl_config()
        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else effective.base_url
        )
        self.token = str(token).strip() if token is not None else effective.token
        self.timeout = (
            float(timeout) if timeout is not None else effective.request_timeout_seconds
        )
        self._client = client

    def _validate(self) -> None:
        if not self.base_url:
            raise RuntimeError("AUTODL_BASE_URL is required")
        if not self.token:
            raise RuntimeError("AUTODL_TOKEN is required")

    @property
    def headers(self) -> dict[str, str]:
        self._validate()
        return {"Authorization": self.token}

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._client is not None:
            response = await self._client.request(
                method, url, headers=self.headers, **kwargs
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method, url, headers=self.headers, **kwargs
                )
        response.raise_for_status()
        return response

    async def submit(
        self, workflow: AutoDLWorkflowSpec, payload: dict[str, Any]
    ) -> str:
        response = await self._request(
            "POST", f"{self.base_url}{workflow.submit_path}", json=payload
        )
        data = response.json()
        task_id = str(data.get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("AutoDL submit response missing task_id")
        return task_id

    async def get_result(
        self, workflow: AutoDLWorkflowSpec, task_id: str
    ) -> AutoDLTaskResult:
        path = workflow.result_path_template.format(task_id=task_id)
        response = await self._request("GET", f"{self.base_url}{path}")
        payload = response.json()
        if str(payload.get("code") or "").lower() != "success":
            raise RuntimeError(str(payload.get("msg") or "AutoDL result query failed"))
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        status = str(data.get("status") or "").strip().lower()
        output_url = ""
        for item in data.get("results") or []:
            if (
                isinstance(item, dict)
                and str(item.get("type") or "").lower() == workflow.output_type
            ):
                output_url = str(item.get("url") or "").strip()
                if output_url:
                    break
        return AutoDLTaskResult(
            task_id=str(data.get("task_id") or task_id),
            status=status,
            output_url=output_url,
            request_id=str(payload.get("request_id") or ""),
            client_id=str(data.get("client_id") or ""),
            message=str(payload.get("msg") or ""),
        )

    async def wait_for_result(
        self,
        workflow: AutoDLWorkflowSpec,
        task_id: str,
        *,
        poll_interval: float = 5.0,
        max_polls: int = 360,
        on_progress: Callable[[float], None] | None = None,
    ) -> AutoDLTaskResult:
        for index in range(max_polls):
            result = await self.get_result(workflow, task_id)
            if result.status in {"completed", "succeeded", "success", "done"}:
                if not result.output_url:
                    raise RuntimeError("AutoDL task completed without a video result")
                return result
            if result.status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(result.message or f"AutoDL task {result.status}")
            if on_progress is not None:
                on_progress(0.2 + (index / max(max_polls, 1)) * 0.7)
            await asyncio.sleep(poll_interval)
        raise TimeoutError("Timed out waiting for AutoDL workflow")

    async def download(self, url: str, output_path: str | Path) -> None:
        # Do not forward the AutoDL token to an arbitrary result/CDN host.
        async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
