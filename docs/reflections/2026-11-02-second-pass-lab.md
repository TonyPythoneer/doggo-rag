# Reflection: trying to make the lfm3 Second Pass better

Date: 2026-11-02. Goal: raise the Second Pass accuracy of lfm3-vl-2b with
retrieval and prompt changes, tested in a self-test loop (`scripts/rag-lab.py`).

## Short version

- Retrieval got much better: the image's own species reached the top 5 for 79
  instead of 55 of 97 non-dog images (hybrid vector + BM25 ranking, species cards).
- A Second Pass that reasons before answering looked like a win on the tuning
  split (+2.06 pp) and lost on the held-out split (−4.04 pp). Without the cards
  it did the same, so the effect was the reasoning, not the knowledge.
- The adopted change keeps the Second Pass prompt and drops reasoning: it fixed
  one wolf (with the new wolf card) and broke nothing on either split.
- A `features: [{description, confidence}]` First Pass broke the JSON for 7 of
  47 images and its confidences were nearly constant, so it was dropped.

## Starting point

The lfm3-vl-2b run at `321ebb1` scored 96.4% in both passes and changed no
answer. Its Second Pass returned the First Pass `is_dog` **and** confidence for
all 196 scored images. For the 97 scored non-dog images, the image's own species
was among the 5 retrieved documents 55 times.

## Method

- **Split.** `dev` = even-numbered images of every species (100), `test` = odd
  (100). The split was fixed before tuning but after seeing the error list: 6 of
  the 7 First Pass errors are in `dev`, so `test` mostly measures whether a
  change breaks correct answers.
- **Cache.** Greedy decoding is deterministic, so model outputs are cached by
  their inputs. The lab baseline reproduced the committed run exactly, and the
  official `make eval` run of #6 reproduced the lab's `dev` numbers exactly.
- **Metrics.** Both pass accuracies on the paired cohort, changed / fixed /
  broken images, failures, and the own-species retrieval rate (labels used only
  for scoring, after inference).

## Experiments on `dev`

Accuracies are on the images where both passes succeeded, so the First Pass
number moves with the failures.

| # | Change | First | Second | Changed | Fixed | Broken | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | HEAD | 93.81% | 93.81% | 0 | 0 | 0 | 3 |
| 1 | Hybrid retrieval (vectors + BM25) and species cards | 93.81% | 94.85% | 1 | 1 | 0 | 3 |
| 2 | New Second Pass prompt, no reasoning (with and without an `animal` key) | 93.81% | 93.81% | 0 | 0 | 0 | 3 |
| 3 | Reasoning in both passes | 97.59% | 97.59% | 0 | 0 | 0 | 17 |
| 4 | Reasoning in the Second Pass only, HEAD prompt, 2048 tokens | 95.70% | 96.77% | 3 | 2 | 1 | 7 |
| 5 | + reasoning prompt, 4096 tokens, dingo eye color fixed | 93.75% | 94.79% | 3 | 2 | 1 | 4 |
| 6 | + budget forcing, dingo coat and setting fixed | 93.81% | 95.88% | 4 | 3 | 1 | 3 |
| 7 | Control: #6 without the retrieved cards | 93.81% | 95.88% | 6 | 4 | 2 | 3 |
| 8 | #6 with a `features` list (description + confidence each) | — | — | — | — | — | 7 of the first 47 |
| 9 | #6 with a dog card of positive marks only, "never by ruling out" | 93.81% | 92.78% | 5 | 2 | 3 | 3 |
| 10 | #1 with the card fixes of #6 and #9, no reasoning, best chunk per document (**adopted**) | 93.81% | 94.85% | 1 | 1 | 0 | 3 |

## The held-out split

| # | First | Second | Changed | Fixed | Broken |
|---|---:|---:|---:|---:|---:|
| 0 HEAD | 98.99% | 98.99% | 0 | 0 | 0 |
| 6 reasoning + cards | 98.99% | 94.95% | 4 | 0 | 4 |
| 7 reasoning, no cards | 98.99% | 94.95% | 4 | 0 | 4 |
| 9 positive dog card | 98.99% | 93.94% | 5 | 0 | 5 |
| 10 **adopted** | 98.99% | 98.99% | 0 | 0 | 0 |

`test` was first scored for #6. Its errors were read to find the cause, so #9
and #10 were not chosen blind on `test`. #10 is #1 (chosen on `dev` before any
`test` score) plus card fixes, and it changed nothing on `test`.

## Lessons

### 1. A same-model second look cannot beat a 96% first look by much

The First Pass was wrong on 7 of 196 images, mostly hard ones: a wolf among
people, a white bear cub on its back, a dog in an engraving, a dog among wolves.
Every reasoning variant changed 3–6 answers per 100 images; #6, #7 and #9
together fixed 9 answers and broke 19 over both splits. With so few errors to
fix, any noise in the second look costs more than it gains.

### 2. The tuning split lied; the held-out split caught it

#6 was +2.06 pp on `dev` and −4.04 pp on `test`. All `test` breaks went the same
way: a correct "not a dog" became "dog". The reasoning showed the mechanism: the
model ruled out each look-alike card by one mismatch ("the coat is not rust, so
not a fox", "the feet are not white, so not a dingo") and the dog card, which
then said "coat of any color, ears floppy or erect", could never be ruled out.
When the right species was not among the candidates, "dog" won by elimination.
Rewriting the dog card to positive marks only and forbidding ruling-out (#9) did
not stop it.

### 3. Without reasoning, the second look is a copy, and that is safe

No prompt wording changed a single answer when the model answered directly
(#2). With an extra `animal` key it even wrote `"animal": "Domestic Dog",
"is_dog": false` for a dog it had called "not a dog" in the First Pass. LFM3-VL
has a reasoning mode that mlx-vlm switches off by default; switched on, the
model wrote "The first pass was wrong (false). So in the second pass, we should
correct to true" — and was right there, and wrong on `test`. The direct Second
Pass changes an answer only when the retrieved text is strong, which happened
once: the new wolf card turned `wolf_008` (a wolf kissing a woman) to "not a dog".

### 4. A small text embedding cannot tell species apart from anatomy words

Descriptions are made of words every species shares ("gray", "erect ears",
"long muzzle"). MiniLM ranked the wolf card above the koala card for "gray and
brown fluffy coat; large round fluffy ears with white fringe; clinging to tree
trunk". Four embedding models and per-phrase queries all stayed below 35%
own-species top-1 for non-dogs. BM25 weights the rare words ("fringe",
"clinging", "mask") and lifted the own species into the top 5 from 55 to 79 of
97 non-dog images. Always keeping the dog card puts the tested hypothesis next
to the closest look-alikes.

### 5. Reasoning in the First Pass is fragile and slow

With reasoning, the First Pass drafted its long JSON description inside the
reasoning, ran past 2048 tokens or broke the JSON for 17 of 100 images, and took
about 30 s instead of 2 s per image.

### 6. Features with confidence did not survive the JSON

The idea (#8): the First Pass lists `features: [{description, confidence}]`,
the joined descriptions stay the retrieval query, and the Second Pass sees each
feature with its confidence so it knows what to re-check. I stopped the run after
47 `dev` dog images: 7 First Pass answers had failed (5 invented an extra key
such as `features_separating_from_other_animals`, 2 closed the object before
`is_dog`), against 1 of the same 47 with the plain description. The confidences
carried little: 372 of 422 were 0.9 or higher. A 2B model writes nested JSON
less reliably than a string, and its per-feature confidence is mostly a
constant, so the change was reverted.

### 7. Reasoning makes knowledge errors visible

The reasoning broke `dingo_004`, a white dingo in a zoo, and showed why: the
dingo card said "amber eyes" and listed no white coat. Both were errors in the
card (dingoes can be all white, and their eyes range from yellow to brown) and
are fixed. The image stayed broken under reasoning: the model then saw "dark
eyes" and still preferred the dog.

### 8. A change tuned on one model can break another

The first official qwen3-vl-2b run with #10 failed 58 Second Passes: qwen3
answered `"is_dog": "true"`, a string the strict schema rejects. On dog_001–020,
main's pipeline failed 3 Second Passes this way; with the cards as context, 10.
Adding "written without quotes" to the prompt made it worse (68 failures in the
first 73 dogs). Sending each document's best-matching chunk, as main does,
brought it back to 2, and gave lfm3 the same answers as the cards. The live MLX
test caught the same failure on `dog_033`.

### 9. Machine state matters for long runs

Several runs crawled at one image every few minutes. The pipeline's own timings
were normal; the Mac was in idle sleep with short maintenance wakes, during
which `time.perf_counter` stops. Long runs now go through `caffeinate -i`.

## Result

`make eval MODEL=lfm3-vl-2b` with the adopted configuration (#10), 200 images,
196 scored, 4 First Pass parse failures (as before):

| | First Pass | Second Pass |
|---|---:|---:|
| Accuracy | 96.43% | 96.94% (+0.51 pp) |
| Precision / Recall / F1 | 0.960 / 0.980 / 0.970 | 0.970 / 0.980 / 0.975 |
| Mean time | 1.91 s | 1.87 s |

1 wrong-to-right (`wolf_008`), 0 right-to-wrong; `dev` +1.03 pp, `test` 0 pp.
The official run reproduced the lab numbers exactly.

The Second Pass prompt was later reordered because lfm2-vl-450m answered "not a
dog" to almost everything (`2026-11-03-lfm2-second-pass.md`); the current results
of all three models are in `2026-11-03-lfm3-second-pass.md`.

## Limits

- One fixed image is within noise (McNemar's exact test: p = 1.0 for 1 fixed, 0 broken).
- The card fixes (dingo eye color, coat, setting; the positive dog card) were
  found through errors on `dev` and, for the dog card, on `test`. They are
  factual, but not chosen blind.
- The dataset gives the lfm3 First Pass only 7 errors; a harder or larger held-out
  set is needed to show a real Second Pass gain.
