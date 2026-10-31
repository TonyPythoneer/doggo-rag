# Dataset Image Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users select evaluation-dataset photos by default and optionally upload an outside image for the same inference pipeline.

**Architecture:** A label-free catalog maps generic IDs to manifest images and serves previews. Validated uploads receive opaque IDs and are stored temporarily. The prediction flow resolves either reference before invoking the existing predictor; the React UI presents dataset thumbnails first and upload as a secondary option.

**Tech Stack:** FastAPI, Pillow, python-multipart, pytest, React, TypeScript, Axios.

**Spec:** `docs/superpowers/specs/2026-10-31-dataset-image-picker-design.md`

## Global Constraints

- Ground-truth labels must not reach inference or RAG retrieval.
- Preserve existing CLI/evaluation path-based image references.
- Do not modify inference prompts, retrieval methodology, or evaluation labels.
- Uploaded files must be validated, limited to 20 MiB, and expire after 24 hours.

## Review Focus

- Missing manifest: return a clear unavailable response.
- Missing image rows: omit unavailable files from the catalog.
- Unknown or traversal IDs: return not found without accessing outside the dataset or upload directory.
- Non-image, malformed, unsupported, and oversized uploads: reject before storing.
- Encoded animal labels in file names: expose only generic names and opaque IDs.

---

### Task 1: Add safe image catalog, preview, and upload APIs

**Files:**
- Modify: `app/dataset.py`
- Create: `app/api/images.py`
- Modify: `app/main.py`, `pyproject.toml`, `uv.lock`
- Test: `tests/test_api.py`

**Interfaces:**
- `list_dataset_images(cfg: AppConfig | None = None) -> list[dict[str, str]]` returns `image_id`, generic `name`, and `preview_url` only.
- `GET /api/v1/images` lists images. `GET /api/v1/images/dataset/{image_id}` serves a dataset preview. `POST /api/v1/images/uploads` accepts multipart `file` and returns `image` plus `preview_url`. `GET /api/v1/images/uploads/{upload_id}` serves a temporary preview.
- `resolve_dataset_image_id(image_id, cfg)` and `resolve_uploaded_image(upload_id)` resolve opaque references within managed directories.

- [x] Test catalog shape and label secrecy, missing files and manifest, valid preview, unknown/traversal IDs, valid upload, malformed and non-image input, unknown upload IDs, and upload size limit.
- [x] Observe API tests fail before implementation.
- [x] Implement manifest-backed ID mapping and safe path containment.
- [x] Validate image content with Pillow; store uploads with UUID names, 20 MiB limit, and 24-hour expiry.
- [x] Register routes and add multipart dependency.
- [x] `uv run pytest tests/test_api.py -q` → 26 passed.

### Task 2: Resolve opaque references in prediction

**Files:**
- Modify: `app/dataset.py`, `app/pipeline/orchestrator.py`
- Test: `tests/test_api.py`

**Interfaces:**
- HTTP prediction accepts `dataset:<image_id>` and `upload:<uuid>` only; direct path inputs remain available to CLI/evaluation callers.
- The API resolves opaque references before inference and returns the opaque reference instead of an absolute server path.

- [x] Add tests for dataset reference, uploaded reference, and unknown reference.
- [x] Observe the reference tests fail before resolution was implemented.
- [x] Resolve both reference prefixes while preserving legacy path resolution.
- [x] Restrict the HTTP endpoint to managed references and preserve the opaque reference in its response.
- [x] Add regression tests for direct path rejection and response path secrecy; observe failures before implementation.
- [x] `uv run pytest tests/test_api.py tests/test_pipeline.py -q` → 32 passed after the security fix.

### Task 3: Make the dataset picker the default frontend flow

**Files:**
- Modify: `app/frontend/src/App.tsx`, `app/frontend/src/index.css`

- [x] Show generic dataset thumbnails and selected-image preview by default.
- [x] Add a secondary external upload control that stores and uses the returned reference.
- [x] Clear stale result/error state when the image source changes.
- [x] `npm --prefix app/frontend run build` → compiled successfully after integrating the picker into the updated two-pass frontend.

### Task 4: Align documentation and progress record

**Files:**
- Modify: `docs/agents/spec.md`, `README.md`

- [x] Describe dataset selection as the default and external upload as an option.
- [x] Record changed files, verification results, limitations, and next work.
- [x] `git diff --check` passed.
