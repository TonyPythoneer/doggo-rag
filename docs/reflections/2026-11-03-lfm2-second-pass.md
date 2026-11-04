# Reflection: why the lfm2 Second Pass said "not a dog" to everything

Date: 2026-11-03. Goal: stop the lfm2-vl-450m Second Pass from losing accuracy
against its own First Pass, without touching labels, retrieval or the First Pass.
Tested with the lab loop (`scripts/rag-lab.py`): tuned on `dev` (even-numbered
images), confirmed on `test` (odd-numbered), then `make eval`.

## Short version

- The Second Pass answered `false` for 173 of 192 scored images (59.4% against a
  75.5% First Pass). An earlier lab run of a Second Pass without the retrieved
  notes was as bad (50.0%), so the knowledge was not the cause; the prompt
  layout was.
- Moving the retrieved notes before the First Pass result and ending the prompt
  with one plain question fixed it: First Pass 75.5% → Second Pass 88.5%
  (+13.0 pp, 26 fixed, 1 broken) in the official run, +14.3 pp on `dev` and
  +11.7 pp on the held-out `test` split.
- The gain cannot be credited to the retrieved knowledge (see "Control").
- The prompt is shared, so lfm3-vl-2b and qwen3-vl-2b changed too; see "Result".

## Starting point

The rerun with the current First Pass prompt (`results/lfm2-…` before this
change, Second Pass prompt with the First Pass result first, then the notes,
then "change the answer when a visible feature contradicts it; otherwise keep it"):

| | First Pass | Second Pass |
|---|---:|---:|
| Accuracy (192 scored) | 75.5% | 59.4% |
| `false` answers | 52 | 173 |

The Second Pass turned 76 of 95 correct dogs into "not a dog". The earlier
+7.5 pp of the first run (75.9% → 83.4%) is not a usable baseline: that First
Pass wrote the same sentence ("An animal with a lean body, pointed ears, a long
muzzle, and a bushy tail.") for most images, copied from an older prompt.

## Experiments on `dev`

98 scored images, First Pass 74.49% in every row. Only the Second Pass prompt
changes; the First Pass outputs and the retrieved notes are the same.

| Variant | Second | Δ pp | Fixed | Broken |
|---|---:|---:|---:|---:|
| Committed prompt (result first, notes, "otherwise keep it") | 59.18% | −15.31 | 23 | 38 |
| Same without the keep and "unlike a dog" lines | 76.53% | +2.04 | 23 | 21 |
| … with an inline answer template `{"is_dog": <true\|false\|null>, …}` | 65.31% | −9.18 | 7 | 16 |
| a: notes first, result last, keep line | 85.71% | +11.22 | 16 | 5 |
| b: notes first, result last, one question, inline template | 90.82% | +16.33 | 21 | 5 |
| d: result first + "check the animal the description names" | 84.69% | +10.20 | 12 | 2 |
| e: a + name check | 77.55% | +3.06 | 21 | 18 |
| f: b + name check | 84.69% | +10.20 | 21 | 11 |
| **g: b with the bulleted keys instead of the template (adopted)** | 88.78% | +14.29 | 14 | 0 |
| h: b + keep line | 86.73% | +12.24 | 21 | 9 |
| i: b with a template that leaves out `null` | 60.20% | −14.29 | 24 | 38 |

## Order of decisions

1. b was chosen on `dev` and scored on `test`: +5.32 pp (19 fixed, 14 broken).
2. b on all 200 lfm3-vl-2b images: −2.19 pp (1 fixed, 5 broken, 4 of them
   `null` answers that the inline template lists).
3. g, h and i were then tried on lfm2 `dev`; g was chosen there (0 broken)
   before its own `test` score. g was therefore not chosen blind: b's `test`
   score and lfm3's all-image result were known.
4. g on `test`: +11.70 pp (12 fixed, 1 broken). The official `make eval` run
   reproduced the lab numbers on both splits exactly.
5. A variant of g with the original question wording ("decide whether the
   visible animal is a domestic dog") was tried after seeing lfm3 break two
   `test` dogs (dog_053, dog_055). It was worse for both models (lfm2 +10.4 pp,
   lfm3 −1.1 pp) and was dropped. No further change was made for those two
   images, since they are `test` images.

## Control: the Second Pass without the notes

| Prompt | Split | Second without notes | Second with notes |
|---|---|---:|---:|
| b | `dev` | 91.30% (+16.30 pp, 6 failed) | 90.82% (+16.33 pp) |
| b | `test` | 88.04% (+10.87 pp, 2 failed) | 81.91% (+5.32 pp) |
| g | all | 95 of 192 Second Passes not JSON | 88.54% (+13.02 pp) |

Without the notes, b did as well or better, and g mostly answered in the
prompt's own text form (`Is_dog: true, confidence = 0.9`) instead of JSON. The
lfm2 gain comes from asking again in a layout the 450M model follows, not from
what the notes say. With g, the notes block is also what keeps the answer in
JSON.

## Lessons

1. **A 450M model answers the end of the prompt.** With the policy, the First
   Pass result and five species notes (four of them about other animals) before
   a list of rules ending in "a dingo is always false", it answered `false`.
   With the notes first and its own description and answer right before the
   question, it kept dogs as dogs and still turned 26 look-alikes to `false`.
2. **Small wording changes swing it by 30 pp.** The inline answer template
   moved it from 77% to 65% in one layout and from 89% to 91% in another;
   leaving `null` out of the template sent it back to 60%.
3. **The description often knows better than the answer.** In `dev`, the First
   Pass called coyotes "a wolf", foxes "a fox" and red pandas "a panda", and
   still answered `true`. Pointing at the name (d, e, f) helped less than the
   plain layout change.
4. **A shared prompt must be checked on every model.** b gained 16 pp for lfm2
   on `dev` and cost lfm3 2 pp.

## Result

Superseded by `2026-11-03-lfm3-second-pass.md` (constrained decoding, policy
wording, species cards). `make eval FRESH=1` per model, 200 images, the same
prompts for all three (digest `e3e9aa0946b4a104`); every official run reproduced
its lab numbers:

| Model | First | Second | Δ pp | Fixed | Broken | Failed | `dev` Δ | `test` Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lfm2-vl-450m | 75.52% | 88.54% | +13.02 | 26 | 1 | 8 | +14.29 | +11.70 |
| qwen3-vl-2b | 81.87% | 94.30% | +12.44 | 32 | 8 | 7 | +11.22 | +13.68 |
| lfm3-vl-2b | 96.72% | 96.17% | −0.55 | 1 | 2 | 17 | +1.08 | −2.22 |

- lfm2: all fixes are look-alikes its First Pass called a dog; the break is
  dog_053 (a dog in an old group photo). McNemar p = 4e-7.
- qwen3: 31 of 32 fixes are First Pass `null` answers that became `false`; all
  8 breaks are dingoes turned into `true`. p = 2e-4. qwen3 was only checked.
- lfm3: fixed wolf_006, broke dog_053 and dog_055 (a dog beside cows). p = 1.0.
  Its 17 failures are First Pass parse errors under the current First Pass
  prompt, unchanged by this work.

## Limits

- `dev` and `test` come from the same 200 images; g was chosen with `test`
  numbers of an earlier variant known (see "Order of decisions").
- The gain is not credited to RAG: the notes-free control matched or beat it
  where it produced JSON.
- 8 of 200 lfm2 First Passes still fail to parse (missing comma, broken
  description); they are recorded as errors and not scored.
