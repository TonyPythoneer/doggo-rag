#!/usr/bin/env python3
"""Build, validate and package the offline `doggo-dataset` image collection.

Dataset contract (see the task specification):

* exactly 200 real photographs, 100 `dog` + 100 `non_dog`
* 10 images for each of ten non-dog species (wolf, fox, coyote, dingo, cat,
  koala, red panda, raccoon, bear, hyena)
* every image is downloaded from Wikimedia Commons with source + license
  metadata preserved, licensed CC0 / Public Domain / CC BY / CC BY-SA
* duplicates (SHA-256) and near-duplicates (perceptual dHash) are rejected and
  replaced by the next valid candidate
* the packaged ZIP is only written after every validation check passes

The script has two modes:

    python scripts/build_doggo_dataset.py            # full build
    python scripts/build_doggo_dataset.py --validate-only

Only `Pillow` and `requests` are required (both are already in the project venv).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import random
import re
import shutil
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

try:
    import requests
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(f"Missing dependency: {exc}. Use the project venv (uv run / .venv).")

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "doggo-dataset"
IMAGES_DIR = DATASET / "images"
META_DIR = DATASET / "metadata"
LICENSES_DIR = DATASET / "LICENSES"
BUILD_DIR = ROOT / ".doggo_build"
ZIP_PATH = DATASET / "doggo-dataset.zip"

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "doggo-dataset-builder/1.0 (offline vision-model evaluation dataset; local research use)"
)

DOG_TARGET = 100
NON_DOG_TARGETS: dict[str, int] = {
    "wolf": 10,
    "fox": 10,
    "coyote": 10,
    "dingo": 10,
    "cat": 10,
    "koala": 10,
    "red_panda": 10,
    "raccoon": 10,
    "bear": 10,
    "hyena": 10,
}
TARGETS: dict[str, int] = {"dog": DOG_TARGET, **NON_DOG_TARGETS}

MIN_DIM = 480  # shortest side of the original photograph
MIN_LONG = 640  # longest side of the original photograph
THUMB_WIDTH = 1280  # download width (thumbnails stay well above model input size)
NEAR_DUP_DISTANCE = 10  # max 64-bit dHash hamming distance for near-duplicates

# Wikimedia Commons root categories per non-dog species.  Verified against the
# live API (some plain-language categories are template redirects and empty).
CATEGORY_ROOTS: dict[str, list[str]] = {
    "wolf": ["Category:Canis lupus"],
    "fox": ["Category:Vulpes vulpes", "Category:Vulpes"],
    "coyote": ["Category:Canis latrans"],
    "dingo": ["Category:Dingoes"],
    "cat": ["Category:Domestic cats", "Category:Felis silvestris catus"],
    "koala": ["Category:Phascolarctos cinereus"],
    "red_panda": ["Category:Ailurus fulgens", "Category:Ailurus fulgens in zoos"],
    "raccoon": ["Category:Procyon lotor"],
    "bear": ["Category:Ursus arctos"],
    "hyena": ["Category:Crocuta crocuta"],
}
DOG_BREED_ROOT = "Category:Dog breeds by name"
DOG_BREEDS_PER_CATEGORY = 2  # keeps breed diversity high
DOG_CANDIDATE_TARGET = 150  # candidate pool for 100 accepted dog images

# Titles that indicate a non-photograph, an artwork, an AI render, or imagery
# that does not show a living animal of the target species.
DENIED_TITLE = re.compile(
    r"\b(map|maps|maps?|chart|charts|diagram|diagrams|graph|graphs|illustration|illustrated|"
    r"drawing|drawings|painting|paintings|sketch|sketches|artwork|engraving|woodcut|"
    r"logo|icons?|emblem|seal|skeleton|skulls?|fossil|silhouette|collage|montage|"
    r"screenshot|infographic|statistics|statue|sculpture|plush|toy|costume|taxidermy|"
    r"specimen|hide|pelt|fur-skin|furskin|meat|carcass|roadkill|hunt|hunted|hunter|"
    r"trap|trapped|trophy|badge|stamp|coin|flag|poster|pattern|ai-generated|aiart|"
    r"midjourney|dall-e|dalle|diffusion|render|cgi|voxel|clipart|cartoon|anime|"
    r"watermark|composite)\b",
    re.IGNORECASE,
)

MANIFEST_COLUMNS = [
    "filename",
    "label",
    "species",
    "source_url",
    "image_url",
    "author",
    "license",
    "license_url",
    "width",
    "height",
    "file_format",
    "verified",
]

_last_api_call = 0.0
_last_download = 0.0
_download_interval = 1.5  # seconds between image downloads; grows when throttled
_rate_limited_until = 0.0  # downloads pause until this timestamp after a 429


def _note_rate_limit(pause: float = 45.0) -> None:
    """The upload host also rate-limits: back off and slow the download loop."""
    global _rate_limited_until, _download_interval
    _rate_limited_until = time.time() + pause
    _download_interval = min(3.0, _download_interval + 0.5)


def _download_wait() -> None:
    global _last_download
    now = time.time()
    wait = max(_last_download + _download_interval - now, _rate_limited_until - now)
    if wait > 0:
        time.sleep(wait)
    _last_download = time.time()


# --------------------------------------------------------------------- HTTP/API
def _pace(interval: float = 1.3) -> None:
    """Keep the Commons API at roughly one request per second (Wikimedia's limit)."""
    global _last_api_call
    wait = _last_api_call + interval - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_api_call = time.time()


def http_get(url: str, params: dict | None = None, tries: int = 6, timeout: int = 90):
    """GET with retry/backoff for rate limits and transient errors."""
    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            response = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout
            )
        except requests.RequestException as exc:  # network hiccup
            last_error = exc
            time.sleep(3 * (attempt + 1))
            continue
        if response.status_code == 429:
            last_error = RuntimeError(f"HTTP 429 rate limited for {url}")
            _note_rate_limit()
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 30.0
            time.sleep(min(90.0, max(20.0, delay)))
            continue
        if response.status_code in (404, 403):
            # permanent for this URL (deleted file, blocked thumbnail) – no point retrying
            raise RuntimeError(f"HTTP {response.status_code} for {response.url}")
        if response.status_code in (503,) or response.status_code >= 500:
            last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = min(90.0, float(retry_after))
            else:
                delay = min(60.0, 5.0 * (2**attempt))
            time.sleep(delay)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code} for {response.url}")
        return response
    raise RuntimeError(f"request failed after {tries} attempts: {url} ({last_error})")


def _is_throttled(payload: dict) -> bool:
    """Distinguish Wikimedia's throttled empty response from a genuinely empty result.

    While the anonymous request limit is exceeded the API answers
    ``{"batchcomplete": ...}`` with no ``query`` block at all -- which is
    indistinguishable from an empty result set.  A tiny canary request tells the
    two apart: if it also comes back without ``query``, we are throttled.
    """
    if "query" in payload or "error" in payload:
        return False
    try:
        canary = http_get(
            API_URL,
            params={"format": "json", "action": "query", "meta": "siteinfo"},
            tries=1,
            timeout=30,
        ).json()
    except (ValueError, RuntimeError, requests.RequestException):
        return True
    return "query" not in canary


def api(params: dict, tries: int = 6) -> dict:
    base = {"format": "json", "action": "query"}
    last: Exception | None = None
    for attempt in range(tries):
        _pace()
        try:
            response = http_get(API_URL, params={**base, **params}, tries=1, timeout=60)
            payload = response.json()
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            last = exc
            time.sleep(min(60.0, 5.0 * (2**attempt)))
            continue
        if "error" in payload:
            last = RuntimeError(payload["error"].get("info", "api error"))
            time.sleep(min(60.0, 5.0 * (2**attempt)))
            continue
        if _is_throttled(payload):
            last = RuntimeError("throttled (empty response to canary)")
            time.sleep(min(60.0, 15.0 * (attempt + 1)))
            continue
        return payload
    raise RuntimeError(f"MediaWiki API failed: {last}")


IMAGE_INFO_PROPS = "url|size|mime|extmetadata"
EXTMETA_FILTER = "LicenseShortName|LicenseUrl|License|Artist|Credit|Attribution|ImageDescription"


def category_files(category: str, limit: int = 100) -> list[dict]:
    """Return imageinfo dicts for files directly inside a category."""
    results: list[dict] = []
    continuation: str | None = None
    while True:
        params = {
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": str(min(limit, 500)),
            "prop": "imageinfo",
            "iiprop": IMAGE_INFO_PROPS,
            "iiurlwidth": str(THUMB_WIDTH),
            "iiextmetadatafilter": EXTMETA_FILTER,
        }
        if continuation:
            params["gcmcontinue"] = continuation
        payload = api(params)
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            if info:
                info["title"] = page["title"]
                results.append(info)
        continuation = payload.get("continue", {}).get("gcmcontinue")
        if not continuation or len(results) >= limit:
            break
    return results[:limit]


def category_subcats(category: str, limit: int = 500) -> list[str]:
    payload = api(
        {
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "subcat",
            "gcmlimit": str(min(limit, 500)),
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    return sorted(page["title"] for page in pages.values())


# ----------------------------------------------------------------- metadata bits
def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_license(
    short: str | None, long_name: str | None, url: str | None
) -> tuple[str, str] | None:
    """Return (license, license_url) only for licenses this dataset may use."""
    name = strip_html(short) or strip_html(long_name)
    if not name:
        return None
    low = name.lower()

    # hard rejections: NC / ND / non-free / unclear terms
    if (
        re.search(r"\bnc\b", low)
        or "noncommercial" in low
        or "non-commercial" in low
        or "non commercial" in low
    ):
        return None
    if re.search(r"\bnd\b", low) or "no derivatives" in low or "no-derivatives" in low:
        return None
    if any(
        token in low
        for token in ("fair use", "non-free", "permission", "review", "unknown", "unclear")
    ):
        return None
    if "gfdl" in low and "cc by" not in low and "cc0" not in low:
        return None

    license_url = strip_html(url)
    if "cc0" in low:
        return ("CC0 1.0", license_url or "https://creativecommons.org/publicdomain/zero/1.0/")
    if (
        "public domain" in low
        or low.startswith("pd")
        or "pdm" in low
        or low in {"pd", "publicdomain"}
    ):
        return (name, license_url or "https://commons.wikimedia.org/wiki/Commons:Public_domain")
    if "cc by" in low or "cc-by" in low:
        if not license_url:
            version = re.search(r"([0-9]+\.[0-9]+)", low)
            flavour = "by-sa" if ("by-sa" in low or "by sa" in low) else "by"
            license_url = (
                f"https://creativecommons.org/licenses/{flavour}/{version.group(1)}/"
                if version
                else f"https://creativecommons.org/licenses/{flavour}/"
            )
        return (name, license_url)
    return None


def candidate_prefilter(info: dict) -> str | None:
    """Reject a candidate from metadata alone; returns a reason or None."""
    title = info.get("title", "")
    filename = title.split(":", 1)[-1]
    mime = info.get("mime", "")
    width = info.get("width") or 0
    height = info.get("height") or 0
    meta = info.get("extmetadata", {}) or {}

    if mime != "image/jpeg":
        return f"not a JPEG photograph ({mime or 'unknown mime'})"
    if width < MIN_LONG or height < MIN_LONG or min(width, height) < MIN_DIM:
        return f"too small ({width}x{height})"
    if DENIED_TITLE.search(filename):
        return "title indicates artwork/map/diagram/AI render/non-animal content"
    if re.search(r"\.(tif|svg|gif)$", filename, re.IGNORECASE):
        return "unsupported format"
    license_pair = normalize_license(
        meta.get("LicenseShortName", {}).get("value"),
        meta.get("License", {}).get("value"),
        meta.get("LicenseUrl", {}).get("value"),
    )
    if license_pair is None:
        raw = strip_html(meta.get("LicenseShortName", {}).get("value")) or "missing"
        return f"license not permitted or unverifiable ({raw})"
    return None


def build_candidate(info: dict, category: str) -> dict:
    meta = info.get("extmetadata", {}) or {}

    def value(key: str) -> str:
        return strip_html((meta.get(key) or {}).get("value"))

    license_pair = normalize_license(
        meta.get("LicenseShortName", {}).get("value"),
        meta.get("License", {}).get("value"),
        meta.get("LicenseUrl", {}).get("value"),
    ) or ("", "")
    artist = value("Artist") or value("Credit") or value("Attribution") or "Unknown"
    # keep the attribution line short enough for a CSV cell
    if len(artist) > 160:
        artist = artist[:157] + "..."
    description_url = info.get("descriptionurl") or (
        "https://commons.wikimedia.org/wiki/" + info["title"].replace(" ", "_")
    )
    download_url = info.get("thumburl") or info.get("url")
    if (info.get("width") or 0) <= THUMB_WIDTH:
        download_url = info.get("url")
    download_url = (download_url or "").split("?", 1)[0]  # drop ?utm_source=… tracking suffix
    return {
        "title": info["title"],
        "category": category,
        "source_url": description_url,
        "image_url": download_url,
        "artist": artist,
        "license": license_pair[0],
        "license_url": license_pair[1],
        "width": info.get("width") or 0,
        "height": info.get("height") or 0,
    }


# ------------------------------------------------------------------ collection
# Categories inside a species tree that never contain a photo of a living
# animal of that species (skins, maps, logos, illustrations, hybrids, etc.).
DENIED_SUBCATEGORY = re.compile(
    r"familiaris|\bdogs?\b|puppies|hybrid|dingo|fur|skin|meat|hide|coat|"
    r"\bmap|distribution|range|logo|icon|skeleton|skull|fossil|illustration|"
    r"painting|drawing|statue|sculpture|toy|plush|artwork|artifact|sign|"
    r"extinct|reconstruction|video|sound|audio|track|flag|stamp|coin|"
    r"in fiction|media|books|fictional|songs|mascot|competition|show|club",
    re.IGNORECASE,
)


def collect_generic_candidates(
    species: str, need: int, max_levels: int = 2, max_calls: int = 14
) -> list[dict]:
    """BFS over the species' Commons categories, keeping pre-filtered candidates."""
    roots = CATEGORY_ROOTS[species]
    candidates: list[dict] = []
    seen: set[str] = set()
    visited: set[str] = set()
    frontier = list(roots)
    calls = 0
    level = 0
    while frontier and level <= max_levels and calls < max_calls:
        next_frontier: list[str] = []
        for category in frontier:
            if category in visited:
                continue
            visited.add(category)
            calls += 1
            for info in category_files(category, limit=80):
                title = info["title"]
                if title in seen:
                    continue
                if candidate_prefilter(info):
                    continue
                seen.add(title)
                candidates.append(build_candidate(info, category))
            if len(candidates) >= need:
                return candidates
            if level < max_levels and calls < max_calls:
                try:
                    subs = category_subcats(category)
                except RuntimeError:
                    subs = []
                next_frontier.extend(
                    s for s in subs if s not in visited and not DENIED_SUBCATEGORY.search(s)
                )
            if calls >= max_calls:
                break
        frontier = next_frontier
        level += 1
    return candidates


DOG_BREED_QUERY = f'deepcat:"{DOG_BREED_ROOT}" filemime:jpeg'


def collect_dog_candidates(need: int = DOG_CANDIDATE_TARGET) -> list[dict]:
    """Sample photos across many breed categories for a diverse dog subset.

    One Commons category request per breed keeps the request rate low and
    avoids the full-text search backend (whose ``deepcat:`` expansion proved
    unreliable); at most ``DOG_BREEDS_PER_CATEGORY`` photos are taken from any
    one breed so the 100 accepted dogs stay breed-diverse.
    """
    rng = random.Random(20261030)
    breeds = category_subcats(DOG_BREED_ROOT, limit=500)
    rng.shuffle(breeds)
    candidates: list[dict] = []
    seen: set[str] = set()
    calls = 0
    for category in breeds:
        if len(candidates) >= need or calls >= 260:
            break
        calls += 1
        files = category_files(category, limit=25)
        kept = 0
        for info in files:
            title = info["title"]
            if title in seen or candidate_prefilter(info):
                continue
            seen.add(title)
            candidates.append(build_candidate(info, category))
            kept += 1
            if kept >= DOG_BREEDS_PER_CATEGORY:
                break
        if kept == 0 and calls < 260:
            # breed pages that only hold sub-categories ("puppies", "in shows", …)
            try:
                subs = [
                    s
                    for s in category_subcats(category, limit=10)
                    if not DENIED_SUBCATEGORY.search(s)
                ]
            except RuntimeError:
                subs = []
            for sub in subs[:3]:
                if len(candidates) >= need or calls >= 260:
                    break
                calls += 1
                for info in category_files(sub, limit=20):
                    title = info["title"]
                    if title in seen or candidate_prefilter(info):
                        continue
                    seen.add(title)
                    candidates.append(build_candidate(info, sub))
                    kept += 1
                    if kept >= DOG_BREEDS_PER_CATEGORY:
                        break
    return candidates


# ------------------------------------------------------------------ image tools
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash_image(image: Image.Image, size: int = 8) -> tuple[int, int]:
    """Two 64-bit difference hashes (LANCZOS and BICUBIC downscale).

    One resampling is noisy: a near-duplicate of the *same photograph* can sit
    at distance 11 with LANCZOS while measuring 5 with BICUBIC, so both are
    computed and the smaller distance is what matters.
    """
    hashes: list[int] = []
    for resample in (Image.Resampling.LANCZOS, Image.Resampling.BICUBIC):
        small = image.convert("L").resize((size + 1, size), resample)
        pixels = list(small.tobytes())
        bits = 0
        for row in range(size):
            for column in range(size):
                left = pixels[row * (size + 1) + column]
                right = pixels[row * (size + 1) + column + 1]
                bits = (bits << 1) | (1 if left > right else 0)
        hashes.append(bits)
    return (hashes[0], hashes[1])


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def perceptual_close(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Near-duplicate when either perceptual hash is within NEAR_DUP_DISTANCE."""
    return min(hamming(left[0], right[0]), hamming(left[1], right[1])) <= NEAR_DUP_DISTANCE


# ------------------------------------------------------------------ build flow
def relative_image_path(species: str, index: int) -> Path:
    stem = f"{species}_{index:03d}.jpg"
    if species == "dog":
        return Path("images") / "dog" / stem
    return Path("images") / "non_dog" / species / stem


class RateLimited(Exception):
    """The CDN answered 429; the caller should switch to a thumbnail URL."""


_prefer_thumbs = False  # set after the first 429: thumb host stays unthrottled


# Commons generates thumbnails only for a fixed set of bucket widths, and the
# API returns the *original* (upload-host) URL when the requested width rounds
# up to the file's own size.  Picking the largest bucket below the original
# width guarantees a real thumb.wikimedia.org URL, which is not rate-limited.
THUMB_BUCKETS = (1280, 1024, 800, 640, 500, 330, 250, 180, 120)


def pick_bucket(width: int) -> int | None:
    for bucket in THUMB_BUCKETS:
        if bucket < width:
            return bucket
    return None


def resolve_image_url(candidate: dict) -> str | None:
    """Trade an upload-host original for a thumb-host thumbnail URL."""
    bucket = pick_bucket(candidate.get("width") or 0)
    if bucket is None:
        return None
    payload = api(
        {
            "titles": candidate["title"],
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": str(bucket),
        }
    )
    for page in payload.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        url = (info.get("thumburl") or "").split("?", 1)[0]
        if "thumb.wikimedia.org" in url:
            return url
    return None


def _fetch_bytes(url: str, timeout: int = 45) -> bytes | None:
    """Download bytes; raise RateLimited on 429 instead of sleeping for minutes."""
    for attempt in range(3):
        _download_wait()
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if response.status_code == 200:
            global _download_interval
            _download_interval = max(1.2, _download_interval - 0.1)
            return response.content
        if response.status_code == 429:
            global _prefer_thumbs
            _prefer_thumbs = True
            _note_rate_limit(pause=0.0)
            raise RateLimited(url)
        if response.status_code in (403, 404):
            return None
        time.sleep(3 * (attempt + 1))
    return None


def download_candidate(candidate: dict) -> tuple[bytes, int, int, int] | None:
    url = candidate["image_url"]
    if _prefer_thumbs and "upload.wikimedia.org" in url:
        thumb_url = resolve_image_url(candidate)
        if thumb_url:
            url = thumb_url
    data = _fetch_bytes(url)
    if data is None or len(data) < 5_000:
        return None
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "JPEG":
            return None
        width, height = image.size
        if min(width, height) < 400 or max(width, height) < 640:
            return None
        dhash = dhash_image(image)
    return data, width, height, dhash


FAILURE_COLUMNS = ["species", "file_title", "source_url", "stage", "reason"]
FAILURE_CSV = META_DIR / "failed_or_replaced.csv"


def log_failure(rows: list[dict], species: str, candidate: dict, stage: str, reason: str) -> None:
    """Record a rejected candidate in memory *and* on disk immediately, so an
    interrupted run still leaves a complete failed/replaced list behind."""
    row = {
        "species": species,
        "file_title": candidate.get("title", ""),
        "source_url": candidate.get("source_url", ""),
        "stage": stage,
        "reason": reason,
    }
    rows.append(row)
    try:
        write_header = not FAILURE_CSV.exists()
        with FAILURE_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        pass  # the in-memory copy still feeds the final report


def build_dataset() -> None:
    for directory in (IMAGES_DIR, META_DIR, LICENSES_DIR, BUILD_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if FAILURE_CSV.exists():
        FAILURE_CSV.unlink()  # fresh run → fresh failed/replaced list

    accepted: list[dict] = []
    failures: list[dict] = []
    seen_sha: set[str] = set()
    perceptual: list[tuple[tuple[int, int], str]] = []  # (dHash pair, relative path)

    def try_accept(species: str, label: str, candidate: dict, dest: Path) -> bool:
        try:
            fetched = download_candidate(candidate)
        except RateLimited:
            # upload host throttled us: switch to the (unthrottled) thumb host
            thumb_url = resolve_image_url(candidate)
            if not thumb_url:
                log_failure(
                    failures,
                    species,
                    candidate,
                    "download",
                    "rate limited and no thumbnail URL available",
                )
                return False
            candidate = {**candidate, "image_url": thumb_url}
            try:
                fetched = download_candidate(candidate)
            except RateLimited:
                log_failure(
                    failures, species, candidate, "download", "rate limited on thumbnail host"
                )
                return False
        except (RuntimeError, requests.RequestException) as exc:
            log_failure(failures, species, candidate, "download", str(exc)[:200])
            return False
        if fetched is None:
            log_failure(
                failures,
                species,
                candidate,
                "download",
                "empty, undecodable or undersized response",
            )
            return False
        data, width, height, dhash = fetched

        digest = sha256_bytes(data)
        if digest in seen_sha:
            log_failure(
                failures,
                species,
                candidate,
                "duplicate",
                "exact SHA-256 duplicate of an accepted image",
            )
            return False
        for existing_hash, existing_path in perceptual:
            if perceptual_close(dhash, existing_hash):
                log_failure(
                    failures,
                    species,
                    candidate,
                    "near-duplicate",
                    f"perceptual hash within distance {NEAR_DUP_DISTANCE} of {existing_path}",
                )
                return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        seen_sha.add(digest)
        perceptual.append((dhash, str(dest.relative_to(DATASET))))
        accepted.append(
            {
                "filename": str(dest.relative_to(DATASET)),
                "label": label,
                "species": species,
                "source_url": candidate["source_url"],
                "image_url": candidate["image_url"],
                "author": candidate["artist"],
                "license": candidate["license"],
                "license_url": candidate["license_url"],
                "width": width,
                "height": height,
                "file_format": "JPG",
                "verified": "yes",
                "sha256": digest,
                "category": candidate["category"],
            }
        )
        return True

    # 1. dogs first (largest, diversity-sensitive pool)
    dog_candidates: list[dict] = []
    for attempt in range(3):
        print(f"== collecting dog candidates (attempt {attempt + 1})", flush=True)
        dog_candidates = collect_dog_candidates()
        print(f"   {len(dog_candidates)} candidates", flush=True)
        if len(dog_candidates) >= DOG_TARGET + 20:
            break
        time.sleep(20)
    count = 0
    for candidate in dog_candidates:
        if count >= DOG_TARGET:
            break
        if try_accept("dog", "dog", candidate, DATASET / relative_image_path("dog", count + 1)):
            count += 1
    if count < DOG_TARGET:
        raise SystemExit(f"dog collection stopped at {count}/{DOG_TARGET}")

    # 2. ten non-dog species, ten images each
    for species, target in NON_DOG_TARGETS.items():
        pool: list[dict] = []
        for attempt in range(3):
            print(f"== collecting {species} candidates (attempt {attempt + 1})", flush=True)
            pool = collect_generic_candidates(species, need=target * 5)
            print(f"   {len(pool)} candidates", flush=True)
            if len(pool) >= target * 3:
                break
            time.sleep(20)
        # interleave categories so a single locality/photographer does not dominate
        by_category: dict[str, list[dict]] = {}
        for candidate in pool:
            by_category.setdefault(candidate["category"], []).append(candidate)
        interleaved: list[dict] = []
        buckets = list(by_category.values())
        index = 0
        while buckets:
            bucket = buckets[index % len(buckets)]
            if bucket:
                interleaved.append(bucket.pop(0))
            buckets = [b for b in buckets if b]
            index += 1
        count = 0
        for candidate in interleaved:
            if count >= target:
                break
            if try_accept(
                species, "non_dog", candidate, DATASET / relative_image_path(species, count + 1)
            ):
                count += 1
        if count < target:
            raise SystemExit(f"{species} collection stopped at {count}/{target}")

    print(f"== accepted {len(accepted)} images", flush=True)
    write_manifest(accepted)
    write_failure_report(failures)
    try:
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        (BUILD_DIR / "accepted.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    except OSError:
        pass  # debug artifact only – never fail the build over it


# ------------------------------------------------------------------- reporting
def write_manifest(rows: list[dict]) -> None:
    with (META_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["filename"]):
            writer.writerow(row)


def write_failure_report(rows: list[dict]) -> None:
    path = META_DIR / "failed_or_replaced.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["species", "file_title", "source_url", "stage", "reason"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"== {len(rows)} failed/replaced candidates logged")


def write_species_summary(counts: dict[str, int], license_counts: dict[str, int]) -> None:
    species_entries = []
    for species, target in TARGETS.items():
        label = "dog" if species == "dog" else "non_dog"
        directory = "images/dog" if species == "dog" else f"images/non_dog/{species}"
        species_entries.append(
            {
                "species": species,
                "label": label,
                "target_count": target,
                "image_count": counts.get(species, 0),
                "directory": directory,
            }
        )
    summary = {
        "dataset": "doggo-dataset",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_images": sum(counts.values()),
        "label_counts": {
            "dog": counts.get("dog", 0),
            "non_dog": sum(v for k, v in counts.items() if k != "dog"),
        },
        "label_definition": (
            "Practical binary definition for this project: domestic dog = dog; "
            "all other selected animals = non_dog. Dingo is explicitly labelled non_dog "
            "for this classification task regardless of biological taxonomy."
        ),
        "species": species_entries,
        "license_counts": license_counts,
        "source": "Wikimedia Commons (https://commons.wikimedia.org)",
    }
    (META_DIR / "species_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_attribution(rows: list[dict]) -> None:
    license_counts: dict[str, int] = {}
    for row in rows:
        license_counts[row["license"]] = license_counts.get(row["license"], 0) + 1

    lines = [
        "# Attribution — doggo-dataset",
        "",
        (
            "Every image in this dataset was downloaded from [Wikimedia Commons]"
            "(https://commons.wikimedia.org) and is reused under a license that permits "
            "free redistribution (CC0, Public Domain, CC BY or CC BY-SA). "
            "Copyright remains with the original photographers and uploaders."
        ),
        "",
        (
            "Reuse requirements: keep the author, source page and license information below "
            "with the images you redistribute, and note that CC BY / CC BY-SA images may only "
            "be adapted under the same licence terms. See:"
        ),
        "",
        "- https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia",
        "- https://creativecommons.org/reusing-cc-licensed-content/",
        "",
        "## Licence totals",
        "",
        "| License | Images |",
        "|---|---:|",
    ]
    for name in sorted(license_counts, key=lambda n: (-license_counts[n], n)):
        lines.append(f"| {name} | {license_counts[name]} |")
    lines += [
        "",
        "## Per-image attribution",
        "",
        "| File | Author / uploader | License | Source page |",
        "|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["filename"]):
        author = row["author"].replace("|", "/")
        lines.append(
            f"| {row['filename']} | {author} | [{row['license']}]({row['license_url']}) "
            f"| {row['source_url']} |"
        )
    lines.append("")
    (LICENSES_DIR / "ATTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ validation
def dhash_file(path: Path) -> int:
    with Image.open(path) as image:
        image.load()
        return dhash_image(image)


def validate(write_reports: bool = True) -> dict:
    """Run the nine validation checks from the specification."""
    manifest: dict[str, dict] = {}
    manifest_path = META_DIR / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                manifest[row["filename"]] = row

    image_files = (
        sorted(p for p in IMAGES_DIR.rglob("*") if p.is_file()) if IMAGES_DIR.exists() else []
    )
    rel_paths = [str(p.relative_to(DATASET)) for p in image_files]

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> bool:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    # 1. exactly 200 image files
    check(
        "exactly_200_images",
        len(image_files) == 200,
        f"{len(image_files)} image files found (expected 200)",
    )

    # 2. 100 dogs
    dog_files = [p for p in rel_paths if p.startswith("images/dog/")]
    check("dog_count_100", len(dog_files) == 100, f"{len(dog_files)} dog images (expected 100)")

    # 3. 10 per non-dog species
    counts: dict[str, int] = {"dog": len(dog_files)}
    per_species_ok = True
    for species, target in NON_DOG_TARGETS.items():
        found = len([p for p in rel_paths if p.startswith(f"images/non_dog/{species}/")])
        counts[species] = found
        if found != target:
            per_species_ok = False
    check(
        "ten_per_non_dog_species",
        per_species_ok,
        ", ".join(f"{s}={counts[s]}" for s in NON_DOG_TARGETS),
    )

    # 4/5. decodable, non-empty
    decode_errors: list[str] = []
    empty_files: list[str] = []
    hashes: dict[str, str] = {}
    dhashes: list[tuple[str, int]] = []
    formats: set[str] = set()
    for path in image_files:
        rel = str(path.relative_to(DATASET))
        size = path.stat().st_size
        if size == 0:
            empty_files.append(rel)
            continue
        try:
            with Image.open(path) as image:
                image.load()
                formats.add((image.format or "").upper())
                dhashes.append((rel, dhash_image(image)))
        except Exception as exc:  # noqa: BLE001 - any decode failure is a validation failure
            decode_errors.append(f"{rel}: {exc}")
    check(
        "all_images_decode",
        not decode_errors,
        "; ".join(decode_errors[:5]) or f"{len(image_files)} images decoded",
    )
    check(
        "no_zero_byte_or_corrupt_files",
        not empty_files and not decode_errors,
        f"zero-byte: {len(empty_files)}, corrupt: {len(decode_errors)}",
    )

    # 6. duplicates: exact SHA-256 and perceptual near-duplicates
    exact_dupes: list[str] = []
    seen: dict[str, str] = {}
    for path in image_files:
        rel = str(path.relative_to(DATASET))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            exact_dupes.append(f"{rel} == {seen[digest]}")
        else:
            seen[digest] = rel
        hashes[rel] = digest
    near_dupes: list[str] = []
    for i in range(len(dhashes)):
        for j in range(i + 1, len(dhashes)):
            if perceptual_close(dhashes[i][1], dhashes[j][1]):
                near_dupes.append(f"{dhashes[i][0]} ~ {dhashes[j][0]}")
    check(
        "no_duplicate_files",
        not exact_dupes,
        "; ".join(exact_dupes[:5]) or "no exact SHA-256 duplicates",
    )
    check(
        "no_near_duplicate_files",
        not near_dupes,
        "; ".join(near_dupes[:5]) or "no perceptual near-duplicates",
    )

    # 7. manifest covers every image, no orphans
    missing = [p for p in rel_paths if p not in manifest]
    orphans = [p for p in manifest if p not in set(rel_paths)]
    check(
        "manifest_matches_images",
        not missing and not orphans,
        f"missing entries: {missing[:5] or 'none'}; orphan entries: {orphans[:5] or 'none'}",
    )

    # 8. traceable source + license for every entry
    bad_meta = []
    for name, row in manifest.items():
        if not row.get("source_url", "").startswith("http"):
            bad_meta.append(f"{name}: source_url")
        if not row.get("image_url", "").startswith("http"):
            bad_meta.append(f"{name}: image_url")
        if not row.get("license"):
            bad_meta.append(f"{name}: license")
        if not row.get("license_url", "").startswith("http"):
            bad_meta.append(f"{name}: license_url")
        if not row.get("author"):
            bad_meta.append(f"{name}: author")
        if row.get("verified", "").lower() not in {"yes", "true", "1"}:
            bad_meta.append(f"{name}: verified")
    check(
        "source_and_license_traceable",
        not bad_meta,
        "; ".join(bad_meta[:5]) or f"{len(manifest)} entries verified",
    )

    # 9. no unexpected files inside the image directories
    unexpected = [
        p
        for p in rel_paths
        if not re.fullmatch(r"images/(dog/dog|non_dog/[a-z_]+/[a-z_]+)_\d{3}\.jpg", p)
    ]
    check(
        "no_unexpected_files_in_image_dirs",
        not unexpected,
        "; ".join(unexpected[:5]) or "only expected .jpg files",
    )

    # label definition sanity: dingo must be non_dog
    dingo_rows = [r for r in manifest.values() if r.get("species") == "dingo"]
    dingo_ok = dingo_rows and all(r.get("label") == "non_dog" for r in dingo_rows)
    check(
        "dingo_labelled_non_dog",
        bool(dingo_ok),
        f"{len(dingo_rows)} dingo rows all labelled non_dog"
        if dingo_ok
        else "dingo rows missing or mislabelled",
    )

    labels_ok = all(
        r.get("label") == ("dog" if r.get("species") == "dog" else "non_dog")
        for r in manifest.values()
    )
    check(
        "label_species_consistency",
        labels_ok and len(manifest) == 200,
        "labels match the binary definition",
    )

    # image formats
    check("jpeg_format_only", formats <= {"JPEG"}, f"formats found: {sorted(formats) or 'none'}")

    # zip integrity, if already built
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            bad = zf.testzip()
            members = [m for m in zf.namelist() if m.endswith(".jpg")]
        check(
            "zip_integrity",
            bad is None,
            f"corrupt member: {bad}" if bad else f"{len(members)} images inside zip",
        )

    report = {
        "dataset": "doggo-dataset",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_images": len(image_files),
        "counts_by_species": counts,
        "label_counts": {
            "dog": counts.get("dog", 0),
            "non_dog": sum(v for k, v in counts.items() if k != "dog"),
        },
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }

    if write_reports:
        (META_DIR / "validation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        lines = [
            "# doggo-dataset validation report",
            "",
            f"Generated: {report['generated_at']}",
            "",
            f"**Overall: {'PASS' if report['all_passed'] else 'FAIL'}**",
            "",
            "## Image count per species",
            "",
            "| Species | Label | Expected | Found |",
            "|---|---|---:|---:|",
        ]
        for species, target in TARGETS.items():
            label = "dog" if species == "dog" else "non_dog"
            lines.append(f"| {species} | {label} | {target} | {counts.get(species, 0)} |")
        lines += [
            f"| **total** | | **200** | **{len(image_files)}** |",
            "",
            "## Checks",
            "",
            "| # | Check | Result | Detail |",
            "|---|---|---|---|",
        ]
        for index, item in enumerate(checks, start=1):
            detail = item["detail"].replace("|", "/")
            lines.append(
                f"| {index} | {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |"
            )
        lines += [
            "",
            "## Files",
            "",
            "- `metadata/manifest.csv` — per-image source and licence metadata",
            "- `metadata/species_summary.json` — counts and label definition",
            "- `metadata/failed_or_replaced.csv` — rejected/replaced candidates and reasons",
            "",
        ]
        (META_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    return report


# ------------------------------------------------------------------- README/zip
def write_readme(counts: dict[str, int], all_passed: bool) -> None:
    rows = "\n".join(
        f"| {'dog' if s == 'dog' else 'non_dog'} | {s} | {TARGETS[s]} | {counts.get(s, 0)} |"
        for s in TARGETS
    )
    readme = f"""# doggo-dataset

A fully offline, licence-clean image dataset for binary classification of
**dog vs non-dog** animal photographs. It is designed as a *fixed evaluation
dataset* for comparing a Vision Model's First Pass with its RAG-assisted Second
Pass on the same images.

## Counts

| Label | Species | Target | Actual |
|---|---|---:|---:|
{rows}
| | **total** | **200** | **{sum(counts.values())}** |

Validation: **{"PASS" if all_passed else "FAIL"}** — see `metadata/validation_report.md`.

## Label definition (important)

This dataset uses a practical binary definition, not biological taxonomy:

- `dog` — domestic dogs only (`images/dog/`).
- `non_dog` — every other selected species (`images/non_dog/<species>/`).

Wolf, fox, coyote and **dingo are explicitly labelled `non_dog`** for this
project's classification task, even though the dingo is closely related to the
domestic dog. Do not silently relabel them from taxonomy.

## Structure

```text
doggo-dataset/
├── images/
│   ├── dog/dog_001.jpg … dog_100.jpg
│   └── non_dog/
│       ├── wolf/  fox/  coyote/  dingo/  cat/
│       ├── koala/  red_panda/  raccoon/  bear/  hyena/
├── metadata/
│   ├── manifest.csv            # per-image source, licence, size
│   ├── species_summary.json    # counts + label definition
│   ├── validation_report.json  # automated validation checks
│   ├── validation_report.md
│   └── failed_or_replaced.csv  # rejected/replaced candidates
├── LICENSES/ATTRIBUTION.md
├── README.md
└── doggo-dataset.zip
```

## Sources and licences

All photographs come from [Wikimedia Commons](https://commons.wikimedia.org)
and are reused under **CC0, Public Domain, CC BY or CC BY-SA** licences only.
Images under NonCommercial or NoDerivatives terms, images with unclear
ownership, artwork, illustrations, maps, diagrams, screenshots and suspected AI
renders were rejected during collection. Full per-image attribution (author,
source page, licence, licence URL) is in `manifest.csv` and
`LICENSES/ATTRIBUTION.md`.

## Image selection

- real JPEG photographs, longest side ≥ {THUMB_WIDTH}px at download time,
  shortest side ≥ {MIN_DIM}px
- the target species is visible and identified by the Commons category it was
  taken from
- exact duplicates (SHA-256) and near-duplicates (dual 64-bit perceptual
  dHash on LANCZOS and BICUBIC downscales, distance ≤ {NEAR_DUP_DISTANCE})
  are rejected and replaced by the next valid candidate
- dog images are sampled across hundreds of breed categories for breed, size,
  coat, pose and environment diversity; each species is sampled across several
  Commons categories to vary camera angle, pose, habitat and lighting

## Intended use

Fixed evaluation set only. DOGGO RAG runs one paired workflow per image:

1. **First Pass:** image → vision model → `dog` / `non_dog` + visual description
2. **RAG-assisted Second Pass:** description → text RAG → the same vision model
   on the same image → final label

Both passes must use the same images and the same vision model. Never
modify image contents or labels to improve model results.

## Rebuilding and re-validating

```bash
uv run python scripts/build_doggo_dataset.py                 # full rebuild (network)
uv run python scripts/build_doggo_dataset.py --validate-only # offline re-validation
```
"""
    (DATASET / "README.md").write_text(readme, encoding="utf-8")


def make_zip() -> dict:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DATASET.rglob("*")):
            if not path.is_file() or path == ZIP_PATH:
                continue
            if path.name in {".DS_Store", "Thumbs.db"}:
                continue
            archive.write(path, Path("doggo-dataset") / path.relative_to(DATASET))
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        image_members = [m for m in archive.namelist() if m.endswith(".jpg")]
        members = archive.namelist()
    if bad:
        raise SystemExit(f"zip corrupt at member: {bad}")
    return {
        "path": str(ZIP_PATH.relative_to(ROOT)),
        "bytes": ZIP_PATH.stat().st_size,
        "members": len(members),
        "image_members": len(image_members),
    }


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only", action="store_true", help="re-run validation on an existing dataset"
    )
    args = parser.parse_args()

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    if not args.validate_only:
        if IMAGES_DIR.exists():
            shutil.rmtree(IMAGES_DIR)
        build_dataset()

    # summary + docs need counts, so compute them after the images exist
    rows: list[dict] = []
    manifest_path = META_DIR / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    license_counts: dict[str, int] = {}
    for row in rows:
        counts[row["species"]] = counts.get(row["species"], 0) + 1
        license_counts[row["license"]] = license_counts.get(row["license"], 0) + 1
    write_species_summary(counts, license_counts)
    if rows:
        write_attribution(rows)

    report = validate(write_reports=True)
    write_readme(report["counts_by_species"], report["all_passed"])

    # README changes the archive contents, so the ZIP is built after every doc is written,
    # then re-validated once more (write_reports=False keeps the archived copy identical).
    zip_info = make_zip()
    final = validate(write_reports=False)

    print(json.dumps({"validation": final, "zip": zip_info}, indent=2))
    return 0 if final["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
