# DOGGO RAG Decisions

Architectural and methodology decisions that are not obvious from the code.
`docs/agents/spec.md` remains the source of truth for required behavior.

## 2026-10-31 — Paired First/Second Pass replaces the baseline-vs-RAG A/B design

- **Decision:** Every image runs one paired workflow (First Pass → retrieval on
  the First Pass description → Second Pass) with the same model. The separate
  vision-only baseline experiment and the `baseline`/`rag`/`both` modes were
  removed from the pipeline, CLI, API and frontend.
- **Why:** `spec.md` §1–§9 define the research comparison as First Pass vs
  Second Pass of the same VLM on the same image, not two independent pipelines.
- **Consequences:** Results use schema `paired`, one JSONL per model and
  dataset version (`results/<model_id>/paired-<model_id>-<version>.jsonl`).
  Legacy `baseline.jsonl`/`rag.jsonl` results are kept on disk but are not
  comparable and are never deleted by `make eval FRESH=1`. A control condition
  that isolates RAG from the "second look" effect is a future experiment
  (`spec.md` §13).

## 2026-10-31 — Per-step statuses in API responses and evaluation records

- **Decision:** `step_statuses` has `first_pass`, `retrieval` and
  `second_pass`, each `not_started`, `completed`, `failed` or `skipped`.
  A failing step is `failed`, later steps are `skipped`; steps never reached
  because setup failed (model selection, image resolution, model loading, or an
  unexpected pipeline exception in the evaluation runner) stay `not_started`.
  The same object is stored in every evaluation record, including the
  `in_progress` checkpoints written after the First Pass and after retrieval.
- **Why:** `spec.md` §2 and §9 require completed outputs to survive later
  failures and failures to be recorded rather than scored as `false`.
- **Consequences:** Overall `status` stays `ok`/`error` for scoring; only
  `ok` records with both passes enter the paired metric cohort.

## 2026-10-31 — `download-models.py --check` reuses the inference resolver

- **Decision:** `--check` calls `app.vision.model.resolve_cached_snapshot`,
  the same local-only `snapshot_download` call the model loader uses; the
  default mode downloads with `snapshot_download(repo_id=...)`.
- **Why:** The previous custom cache walker inspected the `snapshots/` parent
  instead of the snapshot, so it reported every cached model as missing, and
  the default mode never downloaded anything.

## 2026-10-31 — `lfm3-vl-2b` uses `LiquidAI/LFM3-VL-2B-MLX-4bit`

- **Decision (confirmed by the project owner):** The default model's repo is
  `LiquidAI/LFM3-VL-2B-MLX-4bit` in `spec.md` §7, `app/config.py` and
  `configs/default.yaml`.
- **Why:** The spec previously named `mlx-community/LFM3-VL-2B-4bit`, which the
  Hub API answered with 401 (not found or private) on 2026-10-31. The LiquidAI
  repo exists, is the checkpoint the owner uses, and is in the local cache.

## 2026-10-31 — Dataset picker groups thumbnails by species

- **Decision:** The dataset picker groups thumbnails under species headings
  (Dogs first, then the other species alphabetically). The catalog returns the
  manifest `species` for each image. The binary `label` (`dog` / `non_dog`) is
  still never sent to the browser, and the species is never sent to inference
  or retrieval. The classification task stays binary (`spec.md` §1, §6).
- **Why:** The owner wanted the picker organized by animal type. Flattening the
  classification target to 11 species was rejected because it changes the
  research question and the metrics. Species stays a display and analysis
  attribute.
- **Consequences:** `spec.md` §11 was amended to allow species headings and
  forbid the binary label in the browser. Images are still addressed only by
  opaque `image-NNNN` IDs when running inference.

## 2026-10-31 — Evaluation resume rules and one model in memory

- **Decision:** A resumed `make eval` skips only images with an `ok` record and
  retries failed ones. If the stored run metadata (prompts, knowledge, retrieval,
  decoding, model repo, dataset version) differs from the current settings, the
  run stops and asks for `make eval FRESH=1`. `get_vision_model` keeps one model
  instance; switching IDs drops the previous one.
- **Why:** Resuming across a prompt or knowledge change silently mixed two
  experiments in one result file. Permanently keeping a transient failure (for
  example a missing model) made a full restart the only fix. AGENTS.md §4 forbids
  holding all models at once, which the per-ID cache did after `make eval`.
- **Consequences:** The latest attempt per image is the stored record; failures
  stay recorded until a retry succeeds. The standalone `/api/v1/vision/*` and
  `/api/v1/rag/retrieve` endpoints were removed: the spec requires only the E2E
  pipeline endpoint, and `/vision/*` accepted arbitrary file paths.

## 2026-10-31 — Python 3.14 minimum and Jinja2 report template

- **Decision:** `requires-python` and the ruff target are now 3.14 (the venv
  already ran 3.14.5; 3.15 stays out because some dependencies do not support
  it). The evaluation HTML report is rendered with Jinja2 from
  `app/evaluation/templates/` (`report.html.j2`, `report.css`); `report.py`
  only prepares the numbers.
- **Why:** The page and its CSS were Python strings with manual escaping.
  Jinja2 was already in `uv.lock` as a transitive dependency; it is now declared
  directly so the report does not depend on an upstream package keeping it.
- **Consequences:** Autoescape is on for every value. The templates are read
  from the source tree (`[tool.uv] package = false`), so they are not packaged.

## 2026-10-31 — Prompt, knowledge and retrieval-text changes after the first full run

- **Decision:** (1) Prompts no longer show example answer values; they name the
  JSON keys only. (2) The First Pass description is requested in a fixed order
  (coat, ears, muzzle, eyes, build, legs, tail, posture, then features that
  separate it from similar animals) and stays the only retrieval query. (3) The
  Second Pass prompt frames the notes as general facts, says "unlike a dog" is
  not evidence for dog, and allows changing the First Pass answer in either
  direction only on a visible contradiction; retrieval scores are no longer
  shown to the model. (4) Chunks are embedded as `title. section: content`.
  (5) Knowledge text no longer mentions dataset labels, and the dingo/dog
  documents no longer argue that dingoes are ambiguous. (6) Result records keep
  `raw_outputs`; the RAG code hash is part of the experiment settings.
- **Why:** In the first run the models copied the example values (79/200 First
  Pass descriptions and 107/200 Second Pass answers were the examples), so
  retrieval and the Second Pass carried little image information.
- **Consequences:** The classification rules and the "description is the query"
  rule are unchanged. Earlier results are not comparable with new runs; rerun
  with `FRESH=1`. Tuning uses the same 200 images, so results are development
  numbers, not held-out ones.

## 2026-10-31 — Tunable values moved into `configs/default.yaml` / `app/config.py`

- **Decision:** `knowledge.chunk_max_chars` (1200), `retrieval.oversample` (3),
  `evaluation.sample_seed` (42) and a `server` section (host, port, CORS
  origins, upload size and age) are now configuration. `SPECIES_DOCUMENT_IDS`,
  `DATASET_SIZE`, `NON_DOG_COUNT` and `RESULT_SCHEMA_VERSION` live in
  `app/config.py` and are imported everywhere else. Scripts read the results
  directory from the config. `Retriever.search` no longer defaults `top_k`.
  The fast-eval dog sample used the decoding seed before; it now has its own
  seed with the same value, so the sampled images are unchanged.
- **Why:** The same values were written in several files and could drift;
  chunk size and oversampling change retrieval but were not recorded.
- **Consequences:** `chunk_max_chars` and `oversample` are part of the result
  metadata, so earlier results refuse to resume (together with the changed RAG
  code hash); rerun with `FRESH=1`. Retrieval behavior with the defaults is
  unchanged. `make` still passes the literal `results` directories to the AI
  log, and the frontend reads `VITE_API_PORT` / `VITE_API_TIMEOUT_MS`.
