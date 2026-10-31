# DOGGO RAG

> Does a RAG-assisted Second Pass make a local Vision Model better at telling
> domestic dogs from look-alike animals (wolf, fox, coyote, dingo, …)?

Each image gets `is_dog`: `true` (domestic dog), `false` (any other animal,
**including dingoes**) or `null` (not enough evidence). Runs fully offline on
Apple Silicon after model download. Full requirements: `docs/agents/spec.md`.

## Results

200 images (100 dogs, 100 look-alikes), same model for both passes:

| Model | First Pass | Second Pass (RAG) | Change | Wrong → Right | Right → Wrong |
|---|---:|---:|---:|---:|---:|
| lfm3-vl-2b | 97.0% | 94.5% | -2.5 pp | 1 | 6 |
| lfm2-vl-450m | 75.9% | 83.4% | +7.5 pp | 20 | 5 |
| qwen3-vl-2b | 93.4% | 93.4% | +0.0 pp | 0 | 0 |

RAG helps the small model and slightly hurts the strongest one.

- Full report: `results/eval.html`. The 3 failed
  runs are listed there and never scored as `false`.
- Lessons from the run: `docs/reflections/`.
- Caveat: the run overlapped a dependency upgrade, and the metadata does not
  record package versions. Rerun `make eval FRESH=1` if exact versions matter.

## How it works

```text
Image -> VLM First Pass -> is_dog + description
                              |
                 description = retrieval query
                              v
          MiniLM embedding -> FAISS -> top-K knowledge docs
                              v
Image + First Pass + docs -> same VLM Second Pass -> is_dog
```

- Same model and decoding (`temperature=0`, `seed=42`) for both passes.
- Labels and file names never reach inference or retrieval.
- Invalid model output is recorded as an error, never guessed.

## Quickstart

```bash
uv sync
make download                 # only step that uses the network
make eval                     # 3 models x 200 images, verify, write report
make eval FRESH=1 LIMIT=5     # quick smoke test
make fast-eval                # all models, 100 non-dogs + 10 sampled dogs -> results/fast/eval.html
make fast-eval MODEL=lfm3-vl-2b DOGS=20   # one model, 20 dogs (see Makefile header for all parameters)
# both write logs/<time>-<LABEL>.md for the AI reviewer, e.g. make fast-eval LABEL=r2
export DOGGO_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
make web                      # web app at http://127.0.0.1:8888 (asks for the key)
```

`make eval` resumes after interruption and refuses to mix runs if prompts,
knowledge or settings changed (then use `FRESH=1`).

If `doggo-dataset/images/` is missing: `python scripts/build_doggo_dataset.py`.

## Single image

```bash
python -m app.cli ingest      # build the FAISS index once
python -m app.cli predict doggo-dataset/images/non_dog/fox/fox_001.jpg [--model qwen3-vl-2b]
```

```bash
curl -s localhost:8888/api/v1/pipeline/predict \
  -H "X-API-Key: $DOGGO_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"image": "dataset:image-0001", "model_id": "lfm3-vl-2b"}'
```

## API

| Endpoint | Responsibility |
|---|---|
| `POST /api/v1/pipeline/predict` | Full two-pass pipeline (main endpoint) |
| `GET /api/v1/images` | Dataset images (IDs, names, species, previews; no binary label) |
| `GET /api/v1/images/dataset/{image_id}` | Dataset image preview |
| `POST /api/v1/images/uploads` | Validate and temporarily store an uploaded image |
| `GET /api/v1/images/uploads/{upload_id}` | Uploaded image preview |
| `GET /api/v1/health` | Model and index readiness |

The prediction response carries both passes, the retrieval query and documents,
per-step statuses (`not_started`, `completed`, `failed`, `skipped`), stage
errors and timings. Outputs of completed steps are kept when a later step fails.
Retrieval `score` and model `confidence` are not calibrated probabilities. The
HTTP endpoint accepts only dataset and upload references; use the CLI for local
paths.

Frontend dev with hot reload: `make serve` + `pnpm --dir frontend dev`.

Access control: `POST /pipeline/predict`, `GET /images` and `POST /images/uploads`
require the `X-API-Key` header to match the server's `DOGGO_API_KEY` (401 otherwise,
503 if the server has no key). Health and image previews stay open; uploads are
unguessable temporary IDs. Serve over HTTPS when not on localhost.

## Tests

```bash
uv run pytest
```

Tests use fake models, so passing tests do **not** prove MLX inference works.

## More

- `docs/agents/spec.md`: requirements, metrics, API, classification rules
- `knowledge/`: species facts. The dog/non-dog rule lives in `configs/prompts/policy.txt`.
