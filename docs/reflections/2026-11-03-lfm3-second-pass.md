# Reflection: making the lfm3 Second Pass at least tie its First Pass

Date: 2026-11-03. Goal (owner): lfm3-vl-2b's Second Pass must at least tie its
First Pass, ideally gain a little. Starting point (`make eval`, digest
`e3e9aa09`): First 96.72% → Second 96.17% (−0.55 pp, 1 fixed, 2 broken) and 17
First Pass failures. Tested with the lab loop on all 200 images, then `dev`
(even) / `test` (odd) where a change was chosen.

## Short version

Three changes, each with its own evidence:

1. **Schema-constrained decoding** (spec §10 prefers it when the runtime has it;
   mlx-vlm does, through llguidance). lfm3's 17 First Pass failures were all one
   shape, `{"description": "...", "true", 0.95}`: keys dropped after the
   description. With decoding held to the Pydantic schema, all 200 parse and the
   other 183 outputs are unchanged.
2. **Policy wording.** "true means a domestic dog only" became "true means the
   animal is a domestic dog", the wording of spec §1. All three answers the lfm3
   Second Pass changed came from "only" in scenes with several subjects.
3. **Species cards as the retrieved text.** The dog document's best-matching
   chunk was nearly always its generic "Facial features" section. With each
   document's "Visual identification" card instead, the lfm3 Second Pass fixed one
   wolf on `dev` and changed nothing on `test`.

## 1. Constrained decoding

Prompt-only fixes, First Pass only, lfm3 failures out of 17 (and lfm2 / qwen3 on
all 200 images where run):

| First Pass variant | lfm3 failures (of the 17) | Other models |
|---|---:|---|
| Current prompt | 17 | — |
| + "Write the key names before their values" | 2 | not run |
| Format line split over lines | 14 | not run |
| Both | 7 | not run |
| `is_dog` and `confidence` before `description` | 0 (and 0 of 200, 97.0%) | lfm2 First Pass 75.5% → 52.6%; qwen3 54 failures |
| **Schema-constrained decoding, description first** | 0 (0 of 200) | see below |

- Decoding follows the schema's field order, so `FirstPassResult` now declares
  `description` first. Asking for the answer before the description cost lfm2
  23 pp of First Pass accuracy.
- mlx-vlm's own helper forces compact JSON. lfm2 writes indented JSON; forced
  into `{"description": "` it rambled past the token limit in 34 of 200 First
  Passes (8 before). The grammar now allows indented JSON with at most 16
  whitespace characters between tokens. Unbounded whitespace let lfm2 fill the
  token limit with blank lines where it wanted to close the object early.
- The grammar cannot finish a description that never ends: lfm2 still fails
  there (10 of 200 in the lab run), recorded as errors as before.

## 2. The policy word "only"

lfm3, current Second Pass prompt, the 8 images either pass got wrong:

| Second Pass variant | Changed | Fixed | Broken |
|---|---:|---:|---:|
| Current | 3 | wolf_006 | dog_053, dog_055 |
| Without the retrieved notes (control) | 2 | — | dog_053, dog_055 |
| "Is the animal described above a domestic dog?" | 3 | wolf_006 | dog_053, dog_055 |
| + "answer for the animal the description is about" | 2 | — | dog_053, dog_055 |
| + "change the answer only when a visible feature contradicts it" | 3 | wolf_006 | dog_053, dog_055 |
| + "people or other animals do not change the answer" | 2 | — | dog_053, dog_055 |
| Earlier prompt (First Pass result first) | 0 | — | — |
| "Is a domestic dog visible in this image?" | 1 | dog_014 | wolf_006 no longer fixed |
| **Policy "the animal is a domestic dog"** | 0 | — | — |

- dog_053 is a dog lying in front of five men in an old photograph, dog_055 dogs
  beside cattle at a trough, wolf_006 a black wolf in front of a crowd. The
  control flipped the two dogs without any notes, so the breaks were not RAG.
- Under the new policy wolf_006 had the same First Pass and the same retrieved
  documents, and the Second Pass kept "dog": its fix had come from the same
  "only", not from the wolf note.
- "Is a domestic dog visible" was not adopted: it asks the Second Pass a
  different question from the First Pass.
- The policy goes into both prompts, so the comparison stays paired. All 200
  lfm3 images: 97.0% = 97.0%, 0 changed answers, 0 failures.

## 3. Cards instead of the best chunk

The best chunk was chosen on 2026-11-02 because card text made qwen3 write
`"is_dog": "true"`; constrained decoding removes that failure. lfm3, new policy:

| Split | First | Second | Fixed | Broken |
|---|---:|---:|---:|---:|
| `dev` (chosen here) | 96.0% | 97.0% | wolf_006 | 0 |
| `test` (scored once afterwards) | 98.0% | 98.0% | 0 | 0 |

The wolf card says "sometimes all white or all black" and "a zoo enclosure";
the black wolf in its enclosure was turned to "not a dog".

Update (2026-11-04): the no-notes control (`make control`) fixed wolf_006 as
well, and gave the same answer as this run on all 200 images, so the fix came
from the second look, not from the wolf card.

## Order of decisions

1. Constrained decoding: chosen from the failure shape, before any accuracy was
   looked at; the whitespace bound from lfm2's failures on all 200 images.
2. Policy: found by reading the three changed answers, two of them `test`
   images. It is justified by spec §1, not by a score, and it did not raise lfm3
   (0 changes).
3. Cards: chosen on lfm3 `dev`, then `test` was scored once. lfm2 and qwen3 were
   only checked in the official run.

## Result

`make eval FRESH=1`, 200 images per model, same prompts and knowledge for all three
(prompts digest `361f64a20b471254`):

| Model | First | Second | Δ pp | Fixed | Broken | Failed | `dev` Δ | `test` Δ | Before (Δ, fixed/broken, failed) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lfm2-vl-450m | 77.37% | 87.89% | +10.53 | 20 | 0 | 10 | +12.50 | +8.51 | +13.02, 26/1, 8 |
| qwen3-vl-2b | 83.67% | 93.88% | +10.20 | 26 | 6 | 4 | +7.07 | +13.40 | +12.44, 32/8, 7 |
| lfm3-vl-2b | 97.00% | 97.50% | +0.50 | 1 | 0 | 0 | +1.00 | 0.00 | −0.55, 1/2, 17 |

The lfm3 run reproduced its lab numbers image for image on both splits; lfm2 and
qwen3 were never run in the lab with the cards.

- lfm3: wolf_006 fixed, nothing broken, nothing failed; p = 1.0.
- lfm2: all 20 fixes are look-alikes its First Pass called a dog; p = 2e-6. Its
  First Pass rose 1.9 pp and its Second Pass fell 0.6 pp, so the gap shrank. The
  lab run with best chunks instead of cards had 21 fixed, 0 broken. Its 10
  failures are descriptions that never end before the token limit (8 before).
- qwen3: 24 of 26 fixes are First Pass `null` answers that became `false`;
  breaks are 5 dingoes and 1 wolf turned into `true` (8 dingoes before);
  p = 5e-4. Failures fell from 7 to 4.
- The lfm2 and qwen3 gains shrank mostly because their First Pass improved (the
  policy wording and constrained decoding apply to both passes).

## Limits

- lfm3's gain is one image (wolf_006); McNemar's exact test cannot separate one
  fixed and zero broken from chance (p = 1.0). The honest claim is "ties, and
  breaks nothing".
- `dev` and `test` are halves of the same 200 images that earlier prompt work
  had already seen; there is still no separate held-out set.
- The policy change and constrained decoding change the First Pass of every
  model too, so the rows are not comparable with earlier runs.
