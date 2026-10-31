#!/usr/bin/env python3
"""Write one Markdown log of an evaluation run that a person can paste to an AI reviewer.

Reads the paired result files of a results directory and adds the git state and the
diff of prompts, knowledge and code, so the reviewer sees what changed and what it did.
Ground-truth labels are used only here, after inference, to explain results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import RESULT_SCHEMA_VERSION, load_config

DIFF_PATHS = ["configs", "knowledge", "app", "scripts", "Makefile"]
MAX_DIFF_LINES = 400
MAX_FLIPS = 12


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def correct(record: dict, key: str) -> bool | None:
    answer = (record.get(key) or {}).get("is_dog")
    return None if answer is None else answer == record["ground_truth_is_dog"]


def mcnemar(better: int, worse: int) -> float:
    n = better + worse
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2**n)


def pct(hits: int, total: int) -> str:
    return f"{hits / total:.1%} ({hits}/{total})" if total else "-"


def model_section(model_dir: Path) -> list[str]:
    files = sorted(model_dir.glob(f"{RESULT_SCHEMA_VERSION}-*.jsonl"))
    if not files:
        return []
    records = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    ok = [r for r in records if r["status"] == "ok"]
    errors = [r for r in records if r["status"] != "ok"]
    meta_file = files[0].with_suffix(".metadata.json")
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    lines = [f"## {model_dir.name}", ""]
    lines.append(
        f"- images: {len(records)} (ok {len(ok)}, errors {len(errors)}); "
        f"prompts {meta.get('prompts_sha256')}, knowledge {meta.get('knowledge_sha256')}, "
        f"top_k {meta.get('retrieval', {}).get('top_k')}"
    )
    for key, name in (("first_pass", "First Pass"), ("second_pass", "Second Pass")):
        hits = sum(correct(r, key) is True for r in ok)
        nondog = [r for r in ok if not r["ground_truth_is_dog"]]
        dog = [r for r in ok if r["ground_truth_is_dog"]]
        lines.append(
            f"- {name}: accuracy {pct(hits, len(ok))}; "
            f"non-dog detected {pct(sum(correct(r, key) is True for r in nondog), len(nondog))}; "
            f"dog detected {pct(sum(correct(r, key) is True for r in dog), len(dog))}; "
            f"null answers {sum((r[key] or {}).get('is_dog') is None for r in ok)}"
        )
    better = [r for r in ok if correct(r, "first_pass") is not True and correct(r, "second_pass")]
    worse = [r for r in ok if correct(r, "first_pass") and correct(r, "second_pass") is not True]
    lines.append(
        f"- wrong->right {len(better)}, right->wrong {len(worse)}, "
        f"McNemar p={mcnemar(len(better), len(worse)):.3f}"
    )
    species = defaultdict(lambda: [0, 0, 0])
    for r in ok:
        row = species[r["species"]]
        row[0] += 1
        row[1] += correct(r, "first_pass") is True
        row[2] += correct(r, "second_pass") is True
    lines += ["", "| species | n | first | second |", "|---|---:|---:|---:|"]
    lines += [f"| {s} | {n} | {a} | {b} |" for s, (n, a, b) in sorted(species.items())]
    descriptions = Counter(r["first_pass"]["description"] for r in ok)
    top_text, top_count = descriptions.most_common(1)[0] if descriptions else ("", 0)
    answers = Counter(
        (r["second_pass"]["is_dog"], r["second_pass"]["confidence"]) for r in ok if r["second_pass"]
    )
    docs = Counter(d for r in ok for d in r["retrieved_document_ids"])
    lines += [
        "",
        f"- unique First Pass descriptions: {len(descriptions)}/{len(ok)}; "
        f"most repeated x{top_count}: {top_text[:120]!r}",
        f"- most common Second Pass (is_dog, confidence): {answers.most_common(3)}",
        f"- most retrieved documents: {docs.most_common(5)}",
    ]
    for title, group in (("right->wrong", worse), ("wrong->right", better)):
        if group:
            lines += ["", f"### {title} (first {MAX_FLIPS})"]
        for r in group[:MAX_FLIPS]:
            lines.append(
                f"- `{r['image_id']}` truth={'dog' if r['ground_truth_is_dog'] else 'not dog'} "
                f"first={r['first_pass']['is_dog']} second={r['second_pass']['is_dog']} "
                f"conf={r['second_pass']['confidence']} docs={r['retrieved_document_ids']}\n"
                f"  query: {r['retrieval_query'][:240]!r}"
            )
    if errors:
        lines += ["", "### errors"]
        lines += [f"- `{r['image_id']}`: {r['error']}" for r in errors[:MAX_FLIPS]]
    return lines + [""]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=None, help="results directory to read (default: config)")
    parser.add_argument(
        "--label", default="run", help="short name of this run, used in the file name"
    )
    parser.add_argument("--console", help="console output file of the run to append")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    args.dir = args.dir or str(load_config().results.resolve_dir())

    now = datetime.now(UTC)
    lines = [
        f"# AI log: {args.label}",
        "",
        f"- time (UTC): {now.isoformat(timespec='seconds')}",
        f"- results dir: {args.dir}; run exit code: {args.exit_code}",
        f"- git: {git('rev-parse', '--short', 'HEAD')} on {git('branch', '--show-current')}",
        "",
        "## Changes since the last commit",
        "",
        "```",
        git("diff", "HEAD", "--stat", "--", *DIFF_PATHS) or "(none)",
        "```",
        "",
    ]
    diff = git("diff", "HEAD", "-U2", "--", *DIFF_PATHS).splitlines()
    if diff:
        lines += ["```diff", *diff[:MAX_DIFF_LINES], "```"]
        if len(diff) > MAX_DIFF_LINES:
            lines.append(f"(diff cut at {MAX_DIFF_LINES} of {len(diff)} lines)")
        lines.append("")
    for model_dir in sorted(p for p in (ROOT / args.dir).iterdir() if p.is_dir()):
        lines += model_section(model_dir)
    if args.console and Path(args.console).is_file():
        tail = Path(args.console).read_text(errors="replace").splitlines()[-40:]
        lines += ["## Console output (last 40 lines)", "", "```", *tail, "```", ""]

    out = ROOT / "logs" / f"{now:%Y%m%d-%H%M%S}-{args.label}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"AI log written: {out.relative_to(ROOT)}  (send this file to the AI reviewer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
