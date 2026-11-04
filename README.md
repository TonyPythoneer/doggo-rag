# DOGGO RAG

> Does a RAG-assisted Second Pass make a local Vision Model better at telling
> domestic dogs from look-alike animals (wolf, fox, coyote, dingo, …)?

Each image gets `is_dog`: `true` (domestic dog), `false` (any other animal,
**including dingoes**) or `null` (not enough evidence). Runs fully offline on
Apple Silicon after model download. Full requirements: `docs/agents/spec.md`.

## Results

200 images (100 dogs, 100 look-alikes), same model and same prompts for both
passes and all three models, hybrid retrieval, schema-constrained decoding
(2026-11-03). "Failed" images are not scored; accuracy is on the images where
both passes succeeded:

| Model | First Pass | Second Pass (RAG) | Change | Wrong → Right | Right → Wrong | Failed |
|---|---:|---:|---:|---:|---:|---:|
| lfm2-vl-450m | 77.4% | 87.9% | +10.5 pp | 20 | 0 | 10 |
| qwen3-vl-2b | 83.7% | 93.9% | +10.2 pp | 26 | 6 | 4 |
| lfm3-vl-2b | 97.0% | 97.5% | +0.5 pp | 1 | 0 | 0 |

What these numbers support:

- **lfm2-vl-450m** gains on both halves of the data: +12.5 pp on the tuning
  half (even-numbered images) and +8.5 pp on the held-out half (odd). All 20
  fixes are look-alikes the First Pass called a dog; no answer was broken.
  McNemar's exact test: p < 0.001.
- **qwen3-vl-2b** (only checked, never used to choose a change) gains +7.1 /
  +13.4 pp on the two halves (p < 0.001), but differently: 24 of its 26 fixes
  are First Pass `null` answers that became "not a dog", and 5 of its 6 breaks
  are dingoes it turned into "dog".
- **For lfm2, the gain comes from the Second Pass prompt layout, not shown to
  come from the knowledge** (no control was run for qwen3). With the First Pass
  result before the retrieved notes, the
  lfm2 Second Pass answered "not a dog" for 173 of 192 images (59.4%). Putting
  the notes first and the First Pass answer right before the question fixed it.
  A Second Pass without the notes did as well where it returned valid JSON
  (`docs/reflections/2026-11-03-lfm2-second-pass.md`).
- **lfm3-vl-2b** ties and breaks nothing: one wolf fixed by the wolf card, on
  the tuning half (+1.0 pp; held-out half 0 pp, no answer changed). One image is
  within noise (p = 1.0); its First Pass is wrong on only 6 of 200 images. The
  earlier −0.5 pp came from the policy wording "a domestic dog only", which made
  it call dogs beside people or cattle "not a dog", with or without the notes
  (`docs/reflections/2026-11-03-lfm3-second-pass.md`).
- **Failures.** Constrained decoding removed lfm3's 17 First Pass failures (keys
  dropped after the description). lfm2's 10 are descriptions that never end
  before the token limit, which a grammar cannot repair.
- **Development numbers.** The changes were chosen on the even-numbered images
  and checked on the odd-numbered ones of the same 200; there is no separate
  held-out set.

Full report: `results/eval.html` (all three models). Failed runs are listed there,
with the raw model text in the JSONL, and never scored as `false`. The per-image JSONL files are the raw results; `results/paired-summary.json`
and `eval.html` are generated from them by `make eval` (rerun instead of editing).

## How it works

```text
Image -> VLM First Pass -> is_dog + description
                              |
                 description = retrieval query
                              v
   MiniLM embedding -> FAISS  +  BM25 keywords -> rank fusion
                              v
   top-K species docs (the dog always among them), card of each
                              v
Image + First Pass + chunks -> same VLM Second Pass -> is_dog
```

- Same model and decoding (`temperature=0`, `seed=42`) for both passes.
- Retrieval fuses vector search with BM25 keyword ranking: the small embedding
  model alone rarely told species apart from anatomy words that every species
  shares. Each species document ends with a "Visual identification" card of
  photo field marks; the Second Pass gets the card of each retrieved species.
- Both passes decode under the JSON Schema of their Pydantic result (llguidance
  in mlx-vlm), so a model cannot drop a key or quote a boolean; the output is
  still validated.
- `vision.second_pass_thinking: true` makes the Second Pass reason before
  answering (LFM3-VL only). It is off: it made lfm3-vl-2b change answers, but on
  held-out images it turned more correct answers wrong than it fixed.
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
export DOGGO_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
make web                      # web app at http://127.0.0.1:8888 (asks for the key)
```

`make eval` resumes after interruption and refuses to mix runs if prompts,
knowledge or settings changed (then use `FRESH=1`).

Run metadata (`results/<model>/*.metadata.json`) records the prompts, knowledge
and inference-code hashes, retrieval and decoding settings, the Hugging Face
snapshot commit of the vision and embedding models, the Python, OS and package
versions, and a dataset version that hashes the manifest and every image.
Any change to these stops a resume.

If `doggo-dataset/images/` is missing: `python scripts/build_doggo_dataset.py`.

## Self-test loop

Try prompt, knowledge or retrieval changes with the lab, not with `make eval`:

```bash
make lab NAME=my-change                # real pipeline on the dev split (even image numbers)
make lab NAME=my-change SPLIT=test     # confirm a finished change on the odd numbers
python scripts/rag-lab.py lfm3 --split test --score results/lfm3-vl-2b/paired-....jsonl
```

It prints both pass accuracies, the fixed and broken images, failures, and how
often the image's own species was retrieved. Model outputs are cached in
`results/lab/cache/`, so a Second Pass change reruns only the Second Pass.
Details: `spec.md` §13.

## Single image

```bash
python -m app.cli ingest      # build the FAISS index (again after editing knowledge/)
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
| `POST /api/v1/images/uploads` | Validate (≤ 20 MiB, ≤ Pillow's ~89 MP limit) and temporarily store an uploaded image |
| `GET /api/v1/images/uploads/{upload_id}` | Uploaded image preview |
| `GET /api/v1/health` | Model and index readiness |

The prediction response carries both passes, the retrieval query and documents,
per-step statuses (`not_started`, `completed`, `failed`, `skipped`), stage
errors and timings. Outputs of completed steps are kept when a later step fails.
Requests run one at a time: a second request waits until the first finishes.
Retrieval fails with a "rebuild" error when the index no longer matches
`knowledge/` or the embedding settings.
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
uv run pytest           # fake models; does not prove MLX inference works
uv run pytest -m live   # real MLX: full pipeline once per model (needs make download + ingest)
```

Live result on 2026-11-03 (schema-constrained decoding): all three models pass.

## More

- `docs/agents/spec.md`: requirements, metrics, API, classification rules
- `knowledge/`: species facts. The dog/non-dog rule lives in `configs/prompts/policy.txt`.
