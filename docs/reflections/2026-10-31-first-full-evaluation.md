# Reflection: the first full paired evaluation

Date: 2026-10-31. Run: `make eval`, 3 models x 200 images, result schema
`paired`, dataset version `1c2d43e84d260763`. The report is
`results/eval.html`. All numbers below come from the
per-image records in `results/*/paired-*.jsonl`.

## What I wanted to learn

Does a second look at the same photo, with the model's first answer plus
animal facts found by RAG, help a small vision model tell domestic dogs from
look-alike animals?

## The headline numbers

| Model | First Pass | Second Pass | Change | Fixed / broken | Failed runs |
|---|---:|---:|---:|---:|---:|
| LFM3 VL 2B | 97.0% | 94.5% | -2.5 pp | 1 / 6 | 0 |
| LFM2 VL 450M | 75.9% | 83.4% | +7.5 pp | 20 / 5 | 1 |
| Qwen3 VL 2B | 93.4% | 93.4% | 0 pp | 0 / 0 | 2 |

The same procedure made one model worse, one better, and did nothing for the
third. My first reading was "RAG helps weak models and hurts strong ones".
Looking at the records, I do not think the data supports any conclusion about
RAG yet. The reasons follow.

## Lesson 1: the prompt example leaked into the experiment

The First Pass prompt shows an example answer, and the models copied it:

| Model | Description is the prompt example word for word | Most common confidences |
|---|---:|---|
| LFM3 VL 2B | 95 / 200 | First 0.92 (89x), Second 0.97 (113x) |
| LFM2 VL 450M | 64 / 199 | First 0.92 (140x), Second 0.97 (116x) |
| Qwen3 VL 2B | 157 / 198 | First 0.92 (198x), Second 0.97 (198x) |

The copied description is "An animal with a lean body, pointed ears, a long
muzzle, and a bushy tail." It is the RAG query, so for those images
retrieval saw the same text whatever the photo showed. 0.92 and 0.97 are the
example confidences in `first_pass.txt` and `second_pass.txt`. Qwen3 never
reported anything else, so its confidence values carry no information.

This was already flagged as an open owner decision after one image. The full
run shows it is not a corner case: for Qwen3 it is the normal behavior. The
experiment measured "how models react to an example in the prompt" at least
as much as "how models react to retrieved knowledge".

What I take from it: an example in a prompt is not neutral. With small
models it becomes the answer. Show the format with placeholders (`<true |
false | null>`, `<0.0-1.0>`, `<what you can see>`) and check how often the
output equals the example before trusting any metric.

## Lesson 2: retrieved knowledge can talk a model out of a correct answer

LFM3 changed 7 answers. 6 went from "not a dog" (correct) to "dog" (wrong):
5 dingoes and 1 wolf. Every one of those Second Pass answers had confidence
0.97, the example value. In those 6 cases:

- The `dingo` document was retrieved only twice in the top 5, never first.
- Two of the dingo images had the copied template description, so retrieval
  returned the template's documents (`bear`, `coyote`, `dog`, ...).
- The descriptions say "canine", "erect ears", "bushy tail": traits that both
  the dog and dingo documents describe.

So the Second Pass got context that blurred the line between dog and dingo
instead of sharpening it, and it switched to "dog". The project rule "dingo
is always false" was in the prompt the whole time. Retrieved text that looks
relevant outweighed a written rule.

What I take from it: RAG context is not automatically helpful. When retrieval
is weak, it adds noise that sounds authoritative. The Second Pass prompt says
"do not blindly accept retrieved knowledge", and LFM3 accepted it anyway.

## Lesson 3: retrieval quality was the weak link

Across all models, `bear` was the top-ranked document far more often than
any other (95x for LFM3, 79x for LFM2, 157x for Qwen3), although only 10 of
200 photos are bears. For dog photos, the `dog` document was in the top 5 only
93 / 100 (LFM3), 59 / 99 (LFM2) and 77 / 100 (Qwen3) times. Much of this
follows from Lesson 1, because the copied description retrieves `bear`
first. A small general-purpose text embedder (`all-MiniLM-L6-v2`) also has
little to separate "pointed ears, long muzzle" between canids.

What I take from it: measure retrieval on its own before judging the whole
pipeline. If the right document is rarely retrieved, a Second Pass result
says little about the value of RAG.

## Lesson 4: a second look can be ignored entirely

Qwen3 gave the same `is_dog` answer in both passes on all 198 scored images.
With its fixed confidences, it looks like the Second Pass anchors on the
First Pass answer shown in the prompt rather than looking at the photo
again. A 0 pp change does not mean "RAG did not matter". It may mean the
model never used it.

What I take from it: report how often answers change, not only accuracy.
The transition counts (fixed / broken / unchanged) showed this immediately;
accuracy alone hid it.

## Lesson 5: keep the raw output when parsing fails

3 runs failed because the First Pass output had no complete JSON object
(`dog_004` for LFM2, `fox_005` and `wolf_001` for Qwen3). They are recorded as
failures, not as wrong answers, which is what the spec requires. But the
stored error keeps only the first 200 characters of the model output, so I
cannot tell whether the answer hit the 512-token limit or the model wandered
off (the LFM2 one starts an encyclopedia entry about dingoes; one Qwen3 one
describes "a dark, curled object"). I first guessed "token limit". The
records cannot confirm that.

What I take from it: a failure record must keep everything needed to explain
the failure. Here that means the full raw text.

## Engineering lessons from getting here

- **Fail loudly when an experiment changes.** Resume used to append results
  from new prompts to old ones in the same file. It now stops and asks for a
  fresh run. Without this, Lesson 1 could not even be fixed cleanly.
- **Small caching bugs cost hours.** The retriever cache was created inside
  the function, so every image reloaded the embedder and the index (2-3 s
  each instead of 0.03 s).
- **Readable reports find problems.** The HTML report's "fixed / broken"
  counts made the Qwen3 and LFM3 behavior obvious. A single accuracy number
  would have hidden both.
- **Record the environment.** A dependency upgrade landed while this run was
  in progress. The run metadata stores prompts, knowledge and decoding, but
  not package versions, so I cannot prove which versions produced the last
  images.

## Lesson 6: "small models gain, big models don't" is not shown yet

My working guess was: RAG helps small VLMs and does not help large ones,
because large ones are already strong enough. The direction of the changes
says something more specific:

| Model | Fixed | Broken | Said "dog", First → Second (true dogs) |
|---|---|---|---|
| LFM2 VL 450M | 20, all non-dogs moved from "dog" to "not a dog" | 4 dogs moved to "not a dog", 1 non-dog moved to "dog" | 137 → 114 (99) |
| LFM3 VL 2B | 1 dog moved to "dog" | 6 non-dogs moved to "dog" | 100 → 107 (100) |
| Qwen3 VL 2B | 0 | 0 | 109 → 109 (100) |

LFM2's First Pass said "dog" far too often (137 times for 99 dogs). Its
Second Pass simply said "dog" less. That fixed 20 non-dogs and broke 4 real
dogs. Every change went the same way, which looks like a shift towards "not
a dog" rather than the model reading the right facts about each animal. The
retrieved documents are mostly about non-dog species, so the extra context
may just make "not a dog" feel more likely.

Why the data cannot confirm the guess yet:

- **Three models, two families, two sizes.** Size is mixed up with model
  family, training and prompt-following habits.
- **Ceiling.** LFM3 scored 97.0% on the First Pass, so it could gain at most
  3 pp but lose up to 97. "No help for strong models" partly follows from
  having almost nothing left to fix.
- **Prompt copying (Lesson 1)** affects the models differently, so the
  comparison is not like for like.
- **No control (spec §13).** Without a Second Pass without RAG, I cannot
  tell "retrieved knowledge helped" from "being asked twice made it more
  careful".

A fairer test: fix the prompts, add a Second Pass without RAG as the
control, score the hard look-alikes (dingo, wolf, coyote, fox) separately,
and compare sizes within one family (for example LFM2-VL 450M vs 1.6B).

## What I would do next

1. Replace the example values in the prompts with placeholders, then run
   `make eval FRESH=1` and compare against this run.
2. Keep the full raw model output in failed records.
3. Evaluate retrieval on its own: for each image, is its own species document
   in the top 5? Try a stronger embedder if the hit rate stays low.
4. Run the control condition from spec §13 (Second Pass without RAG). It
   separates "looking twice" from "looking twice with knowledge".
5. Record package versions in the run metadata.

## Bottom line

The pipeline works end to end and the evaluation is honest: failures are
kept, both passes are scored on the same images, and every number can be
traced to a record. The research answer is not in yet. This run mainly
showed that the prompt example and weak retrieval shape the results more
than the knowledge itself. That is a useful thing to learn before running
the experiment that matters.
