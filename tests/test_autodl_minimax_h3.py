from __future__ import annotations


import httpx
import pytest

from novelvideo.freezone.video_node import (
    get_freezone_video_model_options,
    resolve_freezone_video_backend,
)
from novelvideo.generators.autodl.client import AutoDLWorkflowClient
from novelvideo.generators.autodl.minimax_h3 import (
    AUTODL_MINIMAX_H3_BACKEND,
    AutoDLMinimaxH3ImageReferenceGenerator,
    minimax_h3_resolution,
)
from novelvideo.generators.autodl.workflows import MINIMAX_H3_IMAGE_REFERENCE
from novelvideo.generators.video_generator import (
    ShotReference,
    VideoBackend,
    VideoGenStatus,
    create_video_generator,
)


@pytest.mark.parametrize(
    ("resolution", "ratio", "expected"),
    [
        ("480p", "9:16", "480p竖"),
        ("480p", "16:9", "480p横"),
        ("768p", "9:16", "768p竖"),
        ("768p", "16:9", "768p横"),
        ("1080p", "9:16", "1080p竖"),
        ("1080p", "16:9", "1080p横"),
    ],
)
def test_minimax_h3_resolution_mapping(resolution, ratio, expected):
    assert minimax_h3_resolution(resolution, ratio) == expected


@pytest.mark.asyncio
async def test_autodl_client_uses_raw_authorization_for_submit_and_result():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "task-1", "status": "QUEUED"})
        return httpx.Response(
            200,
            json={
                "code": "Success",
                "data": {
                    "status": "completed",
                    "task_id": "task-1",
                    "results": [
                        {"url": "https://cdn.example/video.mp4", "type": "video"}
                    ],
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = AutoDLWorkflowClient(
            base_url="https://autodl.example",
            token="plain-token",
            client=http_client,
        )
        task_id = await client.submit(MINIMAX_H3_IMAGE_REFERENCE, {"prompt": "move"})
        result = await client.get_result(MINIMAX_H3_IMAGE_REFERENCE, task_id)

    assert result.output_url == "https://cdn.example/video.mp4"
    assert [request.headers["Authorization"] for request in requests] == [
        "plain-token",
        "plain-token",
    ]
    assert requests[0].url.path.endswith("/minimax_h3_lightx2v_v5")
    assert requests[1].url.path.endswith("/result/task-1")


class _FakeClient:
    def __init__(self):
        self.payload = None

    async def submit(self, workflow, payload):
        self.payload = payload
        return "task-1"

    async def wait_for_result(self, workflow, task_id, **kwargs):
        from novelvideo.generators.autodl.client import AutoDLTaskResult

        return AutoDLTaskResult(
            task_id=task_id, status="completed", output_url="https://cdn/video.mp4"
        )

    async def download(self, url, output_path):
        from pathlib import Path

        Path(output_path).write_bytes(b"video")


@pytest.mark.asyncio
async def test_generator_maps_nine_reference_images(monkeypatch, tmp_path):
    fake = _FakeClient()
    monkeypatch.setattr(
        "novelvideo.storage.media_relay.upload_image_file",
        lambda path, ttl: f"https://oss.example/{path.name}",
    )
    references = []
    for index in range(9):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"image")
        references.append(ShotReference("image", str(path), "图片参考"))
    output = tmp_path / "out.mp4"
    generator = AutoDLMinimaxH3ImageReferenceGenerator(client=fake, resolution="768p")
    result = await generator.generate(
        None,
        "camera pans right",
        str(output),
        aspect_ratio="9:16",
        references=references,
    )

    assert result.status == VideoGenStatus.DONE
    assert fake.payload["resolution"] == "768p竖"
    assert [fake.payload[f"ref_image_{index}"] for index in range(9)] == [
        f"https://oss.example/{index}.png" for index in range(9)
    ]
    assert "ref_image_9" not in fake.payload


def test_factory_and_freezone_catalog_keep_autodl_separate_from_newapi():
    generator = create_video_generator(
        AUTODL_MINIMAX_H3_BACKEND,
        client=_FakeClient(),
    )
    assert isinstance(generator, AutoDLMinimaxH3ImageReferenceGenerator)
    assert resolve_freezone_video_backend(
        AUTODL_MINIMAX_H3_BACKEND
    ) == (AUTODL_MINIMAX_H3_BACKEND)
    option = next(
        item
        for item in get_freezone_video_model_options()
        if item["id"] == AUTODL_MINIMAX_H3_BACKEND
    )
    assert option["providerId"] == "autodl"
    assert option["supportedModes"] == ["image_reference"]
    assert option["referenceImageMax"] == 9


def test_autodl_backend_has_explicit_billing_model():
    from novelvideo.api.routes.model_credits import _video_backend_cost_model

    backend = VideoBackend(AUTODL_MINIMAX_H3_BACKEND)

    assert backend is VideoBackend.AUTODL_MINIMAX_H3
    assert (
        _video_backend_cost_model(AUTODL_MINIMAX_H3_BACKEND)
        == backend.value
    )


def test_autodl_database_config_takes_precedence_over_environment(
    monkeypatch, tmp_path
):
    from novelvideo import config
    from novelvideo import model_gateway_settings as settings

    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "_uses_ce_gateway_settings", lambda: True)
    monkeypatch.setenv("AUTODL_BASE_URL", "https://env.example")
    monkeypatch.setenv("AUTODL_TOKEN", "env-token")
    settings.save_autodl_config(
        base_url="https://database.example/",
        token="database-token",
        request_timeout_seconds=90,
    )

    effective = settings.get_effective_autodl_config()

    assert effective.source == "database"
    assert effective.base_url == "https://database.example"
    assert effective.token == "database-token"
    assert effective.request_timeout_seconds == 90
