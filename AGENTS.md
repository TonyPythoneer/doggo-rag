# AGENTS.md

## 1. Project Overview

DOGGO RAG asks: **does a RAG-assisted Second Pass improve the classification accuracy of the same VLM compared with its First Pass?** The First Pass description is the retrieval query; the Second Pass re-inspects the original image with the retrieved knowledge.

The complete system requirements and E2E workflow are defined in `docs/agents/spec.md`, the primary reference for expected system behavior.

## 2. Required Reading

Before making changes, read:

1. `AGENTS.md`
2. `docs/agents/spec.md`
3. `docs/agents/PROGRESS.md`, if present (local only, see §5)
4. `docs/agents/DECISIONS.md`

Also inspect the relevant source code and existing tests before modifying anything.

Do not assume that the existing implementation matches the specification.

## 3. Development Rules

### Follow the Specification

- Treat `docs/agents/spec.md` as the source of truth for product behavior and E2E requirements.
- Do not introduce features or workflows that are not required by the specification.
- If the existing implementation conflicts with the specification, document the conflict and propose a focused fix.
- Do not silently change research methodology or classification rules.

### Keep Changes Small

- Work on one clearly defined task at a time.
- Prefer small, testable changes over large refactors.
- Avoid unnecessary abstractions, frameworks, dependencies, and architectural complexity.
- Preserve working components unless there is a clear reason to change them.

### Verify Before Reporting

- Run relevant tests after making changes.
- Add tests for new behavior and bug fixes.
- Never claim that a test passed unless it was actually executed.
- Clearly distinguish between verified behavior, unverified assumptions, and known limitations.
- Do not claim that MLX model inference works without executing an actual inference test.

### Protect Evaluation Integrity

- Never use ground-truth labels during inference or RAG retrieval.
- Never modify evaluation labels to improve reported accuracy.
- Keep First Pass and Second Pass results separately.
- Use the same selected VLM for both passes.
- Preserve per-image results so that accuracy improvements can be independently verified.
- Do not silently treat inference errors as incorrect classifications.

## 4. Model Management

The supported VLMs and the default are listed in `docs/agents/spec.md` §7.

- Use the same selected model for both inference passes.
- Do not automatically switch models or use silent fallbacks.
- Do not load all supported models simultaneously by default.
- Verify actual runtime compatibility before declaring a model supported.

## 5. Session Management

For multi-session development:

- Read the current project progress before starting.
- Check Git status and existing changes.
- Avoid overwriting unrelated user modifications.
- Keep `docs/agents/PROGRESS.md` updated after each meaningful task: current state, open issues and the next recommended task. Remove issues once resolved; the history of completed work lives in Git commit messages.
- `docs/agents/PROGRESS.md` is a local working file and is ignored by Git. Never commit it, and do not link to it from committed files. Put anything other people need in `README.md`, `docs/agents/DECISIONS.md` or commit messages.

If the implementation cannot be completed in one session, leave the repository in a clear and recoverable state.

## 6. Documentation

Keep documentation aligned with the actual implementation.

Write all code comments, docstrings, user-facing messages and documentation in English.
Source data quoted verbatim (for example dataset file names and attributions) stays as is.

When changing system behavior:

- Update the relevant specification or technical documentation.
- Record important architectural decisions.
- Remove outdated instructions when they are no longer valid.
- Do not document planned behavior as implemented behavior.

## 7. Completion Criteria

A task is complete only when:

1. The implementation matches the relevant requirements.
2. Relevant tests have been executed.
3. The results have been reviewed.
4. Documentation and the local progress file have been updated.
5. Remaining limitations are clearly documented.
