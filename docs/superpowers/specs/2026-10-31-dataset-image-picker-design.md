# Dataset image picker design

## Intent and goals

Make evaluation-dataset photos the default image source in the web UI, while preserving an option to upload an external image. Both sources run through the existing selected-model pipeline. The browser receives opaque IDs and previews only; ground-truth labels and manifest metadata remain server-side. External uploads stay temporary and outside the evaluation dataset.

## Non-goals and constraints

- Do not change inference prompts, labels, retrieval, evaluation, or model selection.
- Keep CLI and evaluation path-based image inputs working.
- Validate uploaded image content, cap size, and expire temporary files.

## Approaches

1. **Dataset image API plus temporary upload API (selected).** A label-free catalog and preview route support dataset selection; a multipart upload route returns an opaque reference to a temporary image. This keeps one inference contract and avoids exposing paths.
2. **Path listing plus multipart inference.** This would mix two input shapes into prediction and reveal server paths, so it was not selected.

## Proposed behavior

- Show dataset image thumbnails by default with generic names and stable references.
- Let the user choose an image or upload an external image (20 MiB maximum, 24-hour retention).
- Resolve both references on the server and send the image path through the existing inference workflow.

## Calls left to the builder

Use the configured manifest as the catalog source. Use opaque stable IDs and generic display labels, do not return labels/species/filenames, and preserve direct path resolution for CLI/evaluation callers. The HTTP prediction endpoint accepts only dataset or upload references.
