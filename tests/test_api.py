"""API contract tests (TestClient, fake model, injected retriever)."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.api.images as images_api
import app.dataset as dataset_api
from app.config import get_config, load_config
from app.main import create_app
from app.pipeline import orchestrator, rag_pipeline
from app.vision import inference
from tests.conftest import FakeVisionModel


class AuthedClient(TestClient):
    """TestClient that sends the API key unless a test overrides the header."""

    def __init__(self, app):
        super().__init__(app, headers={"X-API-Key": "test-key"})


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setattr(get_config().server, "api_key", "test-key")


@pytest.fixture()
def client(monkeypatch, tmp_index):
    fake = FakeVisionModel()
    monkeypatch.setattr(inference, "get_vision_model", lambda *args, **kwargs: fake)
    monkeypatch.setattr(orchestrator, "get_vision_model", lambda *args, **kwargs: fake)
    monkeypatch.setattr(rag_pipeline, "get_retriever", lambda: tmp_index)
    return AuthedClient(create_app())


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_ready"] is True


def test_pipeline_predict_returns_paired_results(client):
    response = client.post("/api/v1/pipeline/predict", json={"image": "dataset:image-0001"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["step_statuses"] == {
        "first_pass": "completed",
        "retrieval": "completed",
        "second_pass": "completed",
    }
    assert body["first_pass"]["is_dog"] is False
    assert body["second_pass"]["is_dog"] is False
    assert body["retrieval_query"] == body["first_pass"]["description"]
    assert body["retrieved_document_ids"]
    assert body["usages"]["first_pass"]["model_id"] == body["usages"]["second_pass"]["model_id"]
    assert "total_s" in body["timings"]


def test_pipeline_predict_rejects_unsupported_model(client):
    response = client.post(
        "/api/v1/pipeline/predict",
        json={"image": "dataset:image-0001", "model_id": "unknown-vlm"},
    )
    assert response.status_code == 422


def test_pipeline_predict_uses_selected_model_for_both_passes(client, monkeypatch):
    requested = []

    def select_model(model_id):
        requested.append(model_id)
        return FakeVisionModel()

    monkeypatch.setattr(orchestrator, "get_vision_model", select_model)
    response = client.post(
        "/api/v1/pipeline/predict",
        json={"image": "dataset:image-0001", "model_id": "qwen3-vl-2b"},
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == "qwen3-vl-2b"
    assert requested == ["qwen3-vl-2b"]
    assert response.json()["usages"]["first_pass"]["model_id"] == "fake-vision"
    assert response.json()["usages"]["second_pass"]["model_id"] == "fake-vision"


def test_pipeline_predict_missing_image(client):
    response = client.post("/api/v1/pipeline/predict", json={"image": "nope.jpg"})
    assert response.status_code == 422


def test_pipeline_api_rejects_direct_filesystem_paths(client, image_catalog):
    image_path, _ = image_catalog

    response = client.post(
        "/api/v1/pipeline/predict",
        json={"image": str(image_path), "model_id": "lfm3-vl-2b"},
    )

    assert response.status_code == 422


def test_pipeline_predict_resolves_dataset_reference(client, image_catalog, monkeypatch):
    image_path, _ = image_catalog
    cfg = load_config()
    cfg.dataset.images_dir = str(image_path.parents[2])
    cfg.dataset.manifest = str(image_path.parents[3] / "manifest.csv")
    fake = FakeVisionModel()
    monkeypatch.setattr(orchestrator, "get_config", lambda: cfg)
    monkeypatch.setattr(dataset_api, "get_config", lambda: cfg)
    monkeypatch.setattr(orchestrator, "get_vision_model", lambda model_id: fake)

    response = client.post(
        "/api/v1/pipeline/predict", json={"image": "dataset:image-0001", "mode": "baseline"}
    )

    assert response.json()["status"] == "ok"
    assert response.json()["image"] == "dataset:image-0001"
    assert fake.calls[0]["image"] == str(image_path)


def test_pipeline_predict_resolves_uploaded_reference(client, image_catalog, monkeypatch):
    _, uploads_dir = image_catalog
    content = BytesIO()
    Image.new("RGB", (2, 2), color="green").save(content, format="PNG")
    uploaded = client.post(
        "/api/v1/images/uploads",
        files={"file": ("outside.png", content.getvalue(), "image/png")},
    ).json()
    fake = FakeVisionModel()
    monkeypatch.setattr(orchestrator, "get_config", load_config)
    monkeypatch.setattr(orchestrator, "get_vision_model", lambda model_id: fake)

    response = client.post(
        "/api/v1/pipeline/predict", json={"image": uploaded["image"], "mode": "baseline"}
    )

    assert response.json()["status"] == "ok"
    assert response.json()["image"] == uploaded["image"]
    assert fake.calls[0]["image"].startswith(str(uploads_dir))


@pytest.mark.parametrize("fail_first", [False, True])
def test_pipeline_removes_uploaded_file_after_success_or_failure(
    client, image_catalog, monkeypatch, fail_first
):
    _, _uploads_dir = image_catalog
    content = BytesIO()
    Image.new("RGB", (2, 2), color="green").save(content, format="PNG")
    uploaded = client.post(
        "/api/v1/images/uploads",
        files={"file": ("outside.png", content.getvalue(), "image/png")},
    ).json()
    fake = FakeVisionModel()
    if fail_first:
        fake.first = {"is_dog": "invalid", "confidence": 0.8, "description": "traits"}
    monkeypatch.setattr(orchestrator, "get_vision_model", lambda model_id: fake)
    from app.api.images import resolve_uploaded_image

    upload_path = resolve_uploaded_image(uploaded["image"].removeprefix("upload:"))

    response = client.post(
        "/api/v1/pipeline/predict", json={"image": uploaded["image"]}
    )

    assert response.status_code == 200
    assert response.json()["status"] == ("error" if fail_first else "ok")
    assert not upload_path.exists()


def test_pipeline_predict_rejects_unknown_image_reference(client):
    response = client.post(
        "/api/v1/pipeline/predict", json={"image": "upload:../../etc/passwd", "mode": "baseline"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert "not found" in response.json()["error"].lower()


@pytest.fixture
def image_catalog(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    image_path = images_dir / "non_dog" / "fox" / "fox_001.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (3, 2), color="red").save(image_path, format="JPEG")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,label,species,license,source_url,author\n"
        "images/non_dog/fox/fox_001.jpg,non_dog,fox,CC0,https://example.test,author\n",
        encoding="utf-8",
    )
    cfg = load_config()
    cfg.dataset.images_dir = str(images_dir)
    cfg.dataset.manifest = str(manifest)
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(images_api, "UPLOAD_DIR", uploads_dir)
    monkeypatch.setattr(images_api, "get_config", lambda: cfg)
    monkeypatch.setattr(dataset_api, "get_config", lambda: cfg)
    return image_path, uploads_dir


def test_dataset_image_catalog_uses_generic_ids_and_hides_binary_label(image_catalog):
    client = AuthedClient(create_app())

    response = client.get("/api/v1/images")

    assert response.status_code == 200
    assert response.json() == [
        {
            "image_id": "image-0001",
            "name": "Image 1",
            "species": "fox",
            "preview_url": "/api/v1/images/dataset/image-0001",
        }
    ]
    assert "non_dog" not in response.text
    assert "label" not in response.text
    assert "filename" not in response.text


def test_dataset_image_preview_serves_manifest_image(image_catalog):
    client = AuthedClient(create_app())

    response = client.get("/api/v1/images/dataset/image-0001")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(response.content)).size == (3, 2)


def test_dataset_image_preview_rejects_unknown_and_traversal_ids(image_catalog):
    client = AuthedClient(create_app())

    assert client.get("/api/v1/images/dataset/unknown").status_code == 404
    assert client.get("/api/v1/images/dataset/%2e%2e%2fmanifest.csv").status_code == 404


def test_dataset_image_catalog_skips_missing_manifest_images(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,label,species,license,source_url,author\n"
        "images/dog/dog_001.jpg,dog,dog,CC0,https://example.test,author\n",
        encoding="utf-8",
    )
    cfg = load_config()
    cfg.dataset.images_dir = str(tmp_path / "images")
    cfg.dataset.manifest = str(manifest)
    monkeypatch.setattr(images_api, "get_config", lambda: cfg)
    monkeypatch.setattr(dataset_api, "get_config", lambda: cfg)

    response = AuthedClient(create_app()).get("/api/v1/images")

    assert response.status_code == 200
    assert response.json() == []


def test_dataset_image_catalog_reports_missing_manifest(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.dataset.manifest = str(tmp_path / "missing.csv")
    monkeypatch.setattr(images_api, "get_config", lambda: cfg)

    response = AuthedClient(create_app()).get("/api/v1/images")

    assert response.status_code == 503
    assert "manifest" in response.json()["detail"].lower()


def test_image_upload_validates_and_serves_image(image_catalog):
    client = AuthedClient(create_app())
    content = BytesIO()
    Image.new("RGB", (4, 3), color="blue").save(content, format="PNG")

    response = client.post(
        "/api/v1/images/uploads",
        files={"file": ("outside.png", content.getvalue(), "image/png")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["image"].startswith("upload:")
    assert body["preview_url"].startswith("/api/v1/images/uploads/")
    preview = client.get(body["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(preview.content)).size == (4, 3)


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "expected_status"),
    [
        ("broken.png", "image/png", b"not an image", 400),
        ("notes.txt", "text/plain", b"plain text", 415),
    ],
)
def test_image_upload_rejects_invalid_image_input(
    image_catalog, filename, content_type, content, expected_status
):
    response = AuthedClient(create_app()).post(
        "/api/v1/images/uploads",
        files={"file": (filename, content, content_type)},
    )

    assert response.status_code == expected_status


def test_image_upload_preview_rejects_unknown_id(image_catalog):
    response = AuthedClient(create_app()).get(
        "/api/v1/images/uploads/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404


def test_image_upload_rejects_files_over_20_mib(image_catalog):
    response = AuthedClient(create_app()).post(
        "/api/v1/images/uploads",
        files={"file": ("large.png", b"x" * (20 * 1024 * 1024 + 1), "image/png")},
    )

    assert response.status_code == 413


@pytest.mark.parametrize("key", ["", "wrong"])
def test_protected_endpoints_reject_bad_key(client, key):
    for method, url in [
        ("post", "/api/v1/pipeline/predict"),
        ("get", "/api/v1/images"),
        ("post", "/api/v1/images/uploads"),
    ]:
        response = getattr(client, method)(url, headers={"X-API-Key": key})
        assert response.status_code == 401, url
    assert client.get("/api/v1/health", headers={"X-API-Key": key}).status_code == 200


def test_protected_endpoints_fail_closed_without_server_key(client, monkeypatch):
    monkeypatch.setattr(get_config().server, "api_key", "")
    assert client.get("/api/v1/images").status_code == 503


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize(
    "limit", ["app.api.images.MAX_UPLOAD_PIXELS", "PIL.Image.MAX_IMAGE_PIXELS"]
)
def test_image_upload_rejects_too_many_pixels(image_catalog, monkeypatch, limit):
    # The first limit hits the upload check, the second Pillow's own bomb error.
    monkeypatch.setattr(limit, 100)
    content = BytesIO()
    Image.new("RGB", (30, 30)).save(content, format="PNG")

    response = AuthedClient(create_app()).post(
        "/api/v1/images/uploads",
        files={"file": ("huge.png", content.getvalue(), "image/png")},
    )

    assert response.status_code == 413


def test_health_does_not_evict_the_loaded_model(client):
    from app.vision import model as vision_model

    loaded = vision_model.get_vision_model("qwen3-vl-2b")
    try:
        assert client.get("/api/v1/health").json()["vision_model_loaded"] is False
        assert vision_model.get_vision_model("qwen3-vl-2b") is loaded
    finally:
        vision_model.reset_model_cache()
