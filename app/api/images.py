"""Label-free dataset image catalog and temporary user image uploads."""

from __future__ import annotations

import re
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.api import require_api_key
from app.config import get_config
from app.dataset import ImageNotFoundError, list_dataset_images, resolve_dataset_image_id

router = APIRouter(prefix="/api/v1/images", tags=["images"])
UPLOAD_DIR = Path(tempfile.gettempdir()) / "doggo-rag-image-uploads"
MAX_UPLOAD_BYTES = get_config().server.max_upload_mb * 1024 * 1024
UPLOAD_MAX_AGE_SECONDS = get_config().server.upload_max_age_hours * 60 * 60
# Pillow's decompression-bomb limit, captured before a library can disable it
# (some mlx_vlm pipelines set it to None).
MAX_UPLOAD_PIXELS = Image.MAX_IMAGE_PIXELS
IMAGE_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
    "GIF": (".gif", "image/gif"),
    "BMP": (".bmp", "image/bmp"),
    "TIFF": (".tiff", "image/tiff"),
}


class UploadResponse(BaseModel):
    image: str
    preview_url: str


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")


def resolve_uploaded_image(upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise ImageNotFoundError(f"Uploaded image not found: {upload_id!r}")
    matches = list(UPLOAD_DIR.glob(f"{upload_id}.*"))
    if len(matches) != 1 or not matches[0].is_file():
        raise ImageNotFoundError(f"Uploaded image not found: {upload_id!r}")
    path = matches[0]
    if time.time() - path.stat().st_mtime > UPLOAD_MAX_AGE_SECONDS:
        path.unlink(missing_ok=True)
        raise ImageNotFoundError(f"Uploaded image expired: {upload_id!r}")
    return path


def remove_uploaded_image(upload_id: str) -> None:
    """Remove one upload by its opaque ID without accepting caller paths."""
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        return
    for path in UPLOAD_DIR.glob(f"{upload_id}.*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def _expire_old_uploads() -> None:
    UPLOAD_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    cutoff = time.time() - UPLOAD_MAX_AGE_SECONDS
    for path in UPLOAD_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


@router.get("", dependencies=[Depends(require_api_key)])
def get_image_catalog() -> list[dict[str, str]]:
    try:
        return list_dataset_images(get_config())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/dataset/{image_id}")
def get_dataset_image(image_id: str) -> FileResponse:
    try:
        path = resolve_dataset_image_id(image_id)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Dataset image not found") from exc
    return FileResponse(path, media_type=_media_type(path))


@router.post(
    "/uploads",
    status_code=201,
    response_model=UploadResponse,
    dependencies=[Depends(require_api_key)],
)
async def upload_image(file: Annotated[UploadFile, File()]) -> UploadResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must use an image content type")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image upload exceeds {MAX_UPLOAD_BYTES // 2**20} MiB")
    try:
        with Image.open(BytesIO(content)) as image:
            image_format = image.format
            if image.width * image.height > MAX_UPLOAD_PIXELS:  # Pillow only warns up to 2x
                raise Image.DecompressionBombError(f"{image.size} is over {MAX_UPLOAD_PIXELS} px")
            image.verify()
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Image dimensions are too large") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    if image_format not in IMAGE_FORMATS:
        raise HTTPException(status_code=415, detail="This image format is not supported")

    _expire_old_uploads()
    upload_id = uuid.uuid4().hex
    suffix, _ = IMAGE_FORMATS[image_format]
    (UPLOAD_DIR / f"{upload_id}{suffix}").write_bytes(content)
    return UploadResponse(
        image=f"upload:{upload_id}",
        preview_url=f"/api/v1/images/uploads/{upload_id}",
    )


@router.get("/uploads/{upload_id}")
def get_uploaded_image(upload_id: str) -> FileResponse:
    try:
        path = resolve_uploaded_image(upload_id)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Uploaded image not found") from exc
    return FileResponse(path, media_type=_media_type(path))
