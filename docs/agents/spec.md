# DOGGO RAG — End-to-End System Specification

## 1. Project Objective

DOGGO RAG is a local-first animal image classification system designed to investigate whether Retrieval-Augmented Generation (RAG) can improve Vision Language Model (VLM) classification accuracy.

The system uses the same selected VLM for two inference passes.

The First Pass performs image classification and generates a visual description.

The RAG component uses that description to retrieve relevant animal knowledge.

The Second Pass analyzes the original image again with the retrieved knowledge and produces the final classification.

### Primary Research Question

Does RAG-assisted second-pass inference improve classification accuracy compared with the same VLM's first-pass inference without RAG?

### Scope

The initial classification task is binary:

- `true`: The animal is a domestic dog.
- `false`: The animal is not a domestic dog.
- `null`: The model cannot make a reliable classification.

Dingoes must always be classified as `false`.

Dog breed classification is outside the scope of this project.

---

## 2. End-to-End Architecture

The required inference workflow is:

```text
Original Image
      |
      v
Selected VLM
      |
      v
Phase 1: First Pass
      |
      |-- is_dog
      |-- confidence
      |-- description
      |
      v
Description
      |
      v
Embedding Model
      |
      v
Vector Retrieval
      |
      v
Animal Knowledge Context
      |
      v
Selected VLM + Original Image
      + First Pass Information
      + Retrieved Knowledge
      |
      v
Phase 2: Second Pass
      |
      |-- is_dog
      |-- confidence
      |
      v
Final Classification
      |
      v
Evaluation and Reporting
```

### Core Requirements

1. The original image must be provided to both inference passes.
2. The same selected VLM must be used for both passes.
3. The First Pass description must be used as the RAG retrieval query.
4. The Second Pass must receive the retrieved knowledge context.
5. The Second Pass must independently reconsider the original image.
6. The Second Pass may change the First Pass classification.
7. Both inference results must be stored separately for evaluation.

The system must not replace the Second Pass with a text-only classification.

The pipeline response must include an individual status for First Pass, RAG retrieval, and Second Pass. Completed steps are `completed`, a failing step is `failed`, later steps that cannot run are `skipped`, and steps not reached because setup failed are `not_started`. Preserve completed outputs when a later step fails. The response also includes `raw_outputs` with the unparsed text returned by the First Pass and Second Pass model calls; a key is absent when that call produced no output.

---

## 3. Phase 1 — First Pass Inference

### Input

- Original animal image.
- Selected VLM.

### Output Schema

```json
{
  "is_dog": true,
  "confidence": 0.92,
  "description": "A medium-sized animal with four legs, pointed ears, a long muzzle, and a bushy tail."
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `is_dog` | boolean or null | Initial dog classification |
| `confidence` | number | Model-reported confidence |
| `description` | string | Visual observations used for RAG retrieval |

### Description Requirements

The description must:

- Focus on visible physical characteristics.
- Describe observable features such as body shape, ears, muzzle, tail, fur, and posture.
- Avoid relying only on the model's initial classification.
- Avoid unsupported species assumptions.
- Provide useful information for retrieving similar animal characteristics.

The description should remain useful even when the initial classification is incorrect.

The First Pass result must be stored before starting RAG retrieval.

---

## 4. RAG — Knowledge Retrieval

### Purpose

Retrieve relevant animal knowledge using the First Pass visual description.

### Retrieval Workflow

1. Receive the First Pass description.
2. Convert the description into an embedding vector.
3. Search the local vector index.
4. Retrieve the most relevant knowledge documents.
5. Return the retrieved documents and retrieval metadata to the inference pipeline.

### Embedding Model

The embedding model is responsible for converting text into vector representations.

It is separate from the VLM.

It must support:

- Knowledge document embedding during index construction.
- Query embedding during retrieval.
- Consistent embedding dimensions and model configuration.

Knowledge embeddings should be generated during ingestion or index construction, not repeatedly during image inference.

### Knowledge Base

The knowledge base should contain factual information about domestic dogs and non-dog animals.

Initial target species:

- Domestic dog
- Wolf
- Fox
- Coyote
- Dingo
- Cat
- Koala
- Red panda
- Raccoon
- Bear
- Hyena

Knowledge documents should focus on distinguishing physical characteristics and species identification.

### Retrieval Requirements

- Use the First Pass description as the retrieval query.
- Do not use ground-truth labels.
- Do not use dataset filenames as retrieval hints.
- Keep the embedding model, knowledge base, chunking strategy, retrieval method, and retrieval settings consistent across supported VLMs.
- Record retrieved document IDs and retrieval scores.
- Keep retrieval separate from final classification.

The retrieval query is expected to vary depending on the First Pass description generated by each VLM.

This is intentional because the project evaluates the complete model-driven RAG workflow.

---

## 5. Phase 2 — Second Pass Inference

### Input

The Second Pass must receive:

1. The original image.
2. The First Pass classification and confidence.
3. The First Pass visual description.
4. The retrieved RAG knowledge context.

The same VLM selected for the First Pass must be used again.

### Output Schema

```json
{
  "is_dog": true,
  "confidence": 0.97
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `is_dog` | boolean or null | Final dog classification |
| `confidence` | number | Model-reported confidence |

### Second Pass Requirements

- Re-examine the original image.
- Consider the First Pass observations.
- Use retrieved knowledge to support species identification.
- Do not blindly accept the First Pass classification.
- Do not blindly accept retrieved knowledge.
- Resolve conflicts by prioritizing visible evidence in the original image.
- Return only the required structured classification result.

The Second Pass does not need to generate another visual description.

---

## 6. Classification Rules

The classification target is domestic dog detection.

| Animal | Expected `is_dog` |
|---|---|
| Domestic dog | `true` |
| Wolf | `false` |
| Fox | `false` |
| Coyote | `false` |
| Dingo | `false` |
| Cat | `false` |
| Koala | `false` |
| Red panda | `false` |
| Raccoon | `false` |
| Bear | `false` |
| Hyena | `false` |
| Uncertain or insufficient evidence | `null` |

A dingo must not be classified as a domestic dog.

The system must not perform dog breed classification.

---

## 7. Supported Vision Models

The system must support the following models:

| Stable ID | Hugging Face Repository | Role |
|---|---|---|
| `lfm3-vl-2b` | `LiquidAI/LFM3-VL-2B-MLX-4bit` | Default |
| `lfm2-vl-450m` | `mlx-community/LFM2-VL-450M-4bit` | Comparison |
| `qwen3-vl-2b` | `mlx-community/Qwen3-VL-2B-Instruct-4bit` | Comparison |

### Model Selection Requirements

- Default model: `lfm3-vl-2b`.
- Allow the user to select any supported model.
- Use the selected model for both inference passes.
- Do not automatically switch to another model.
- Do not silently fall back to a different model.
- Load only the selected model by default.
- Verify actual MLX runtime compatibility.

Each model must be evaluated independently using the same dataset.

---

## 8. Dataset Requirements

The initial evaluation dataset contains 200 real animal photographs.

| Category | Image Count |
|---|---:|
| Domestic dog | 100 |
| Wolf | 10 |
| Fox | 10 |
| Coyote | 10 |
| Dingo | 10 |
| Cat | 10 |
| Koala | 10 |
| Red panda | 10 |
| Raccoon | 10 |
| Bear | 10 |
| Hyena | 10 |
| **Total** | **200** |

### Dataset Rules

- Use real photographs.
- Preserve image source URLs.
- Preserve license information.
- Accept CC0, Public Domain, CC BY, and CC BY-SA licenses.
- Avoid unclear licenses, watermarks, and restricted images.
- Keep ground-truth labels separate from inference inputs.
- Never expose ground-truth labels to the VLM or RAG retrieval process.

The dataset must use stable image identifiers so results can be compared across models.

---

## 9. Evaluation and Reporting

### Evaluation Principle

For every image, compare the First Pass and Second Pass results against the same ground-truth label.

The comparison must be performed independently for each selected VLM.

### Required Metrics

For each model, report:

1. First Pass Accuracy.
2. Second Pass Accuracy.
3. Accuracy Improvement in percentage points.
4. Incorrect-to-Correct transitions.
5. Correct-to-Incorrect transitions.
6. Unchanged correct predictions.
7. Unchanged incorrect predictions.
8. Per-species accuracy.
9. Confusion matrix.
10. Precision, Recall, and F1-score where applicable.
11. Inference time for each pass.

### Accuracy Improvement

```text
Accuracy Improvement =
Second Pass Accuracy - First Pass Accuracy
```

Example:

```text
First Pass Accuracy: 78%
Second Pass Accuracy: 86%

Accuracy Improvement: +8 percentage points
```

These numbers are illustrative only.

### Paired Result Tracking

Every image evaluation must preserve:

- Image ID
- Ground-truth label
- Model ID
- First Pass output
- Second Pass output
- Retrieved document IDs
- Retrieval scores
- First Pass inference duration
- Retrieval duration
- Second Pass inference duration
- Error information, if any

Results from different models must never overwrite or reuse each other's records.

An inference failure must be recorded as an error, not silently converted to `false`.

---

## 10. Error Handling and Data Integrity

The system must handle:

- Invalid image files.
- Unsupported image formats.
- Model loading failures.
- MLX inference failures.
- Invalid JSON responses.
- Embedding failures.
- Empty retrieval results.
- Vector index loading failures.
- Second Pass failures.

Malformed model output must not silently become a valid classification.

Use structured validation, preferably Pydantic, for both inference outputs.

If JSON Schema constrained decoding is unavailable in the actual MLX runtime, use a JSON-only prompt followed by backend validation.

Keep failed records available for debugging and evaluation reporting.

---

## 11. Backend and Frontend Requirements

### Backend

The backend must provide an E2E inference workflow that:

1. Accepts an image.
2. Accepts or uses the selected VLM configuration.
3. Executes the First Pass.
4. Executes RAG retrieval.
5. Executes the Second Pass.
6. Returns both inference results and retrieval metadata.

The API must clearly distinguish First Pass and Second Pass outputs.

### Frontend

Provide a simple interface that allows users to:

- Select an animal image from the existing evaluation dataset (default source).
- Optionally upload an external animal image.
- Select a supported VLM.
- Start inference.
- View the original image.
- View First Pass classification, confidence, and description.
- View retrieved RAG knowledge.
- View Second Pass classification and confidence.
- View the status of First Pass, RAG retrieval, and Second Pass, including the failed step and skipped downstream steps.
- Show each step's duration beside its status, and let the user click a step in the pipeline diagram to focus its result card.
- Show each step's raw output in a collapsible panel with a text search. For RAG retrieval, the raw output is the query and retrieved documents as JSON.
- Compare both results.

Dataset selection may show each image's species as a group heading (for example "Dogs", "Cats", "Foxes"). It must not expose the binary dog / non-dog label, or any other ground-truth label, to the browser, and the species must not be sent to the VLM or RAG retrieval process. The dataset image choices are otherwise referenced only by opaque image IDs. The binary classification task itself is unchanged. External uploads are temporary and are not added to the evaluation dataset. The frontend should focus on demonstrating the E2E workflow rather than adding unnecessary features.

---

## 12. Acceptance Criteria

The initial implementation is considered complete when:

1. A user can select a dataset image or upload an external image.
2. A user can select one of the three supported VLMs.
3. The selected VLM successfully performs the First Pass.
4. The First Pass returns valid JSON containing `is_dog`, `confidence`, and `description`.
5. The description is used as the RAG retrieval query.
6. The embedding model successfully generates the query vector.
7. The vector index returns relevant knowledge documents.
8. The same VLM receives the original image and retrieved context for the Second Pass.
9. The Second Pass returns valid JSON containing `is_dog` and `confidence`.
10. Both inference results are stored independently.
11. The evaluation runner can process the complete 200-image dataset.
12. Phase 1 and Phase 2 accuracy can be compared for each model.
13. Results are isolated by model and dataset version.
14. The frontend can demonstrate the complete E2E workflow.
15. Tests and actual MLX inference have been executed and their results documented.

---

## 13. Research Integrity and Future Experiments

The primary experiment compares:

- First Pass: Original image without retrieved RAG knowledge.
- Second Pass: Original image with First Pass information and retrieved RAG knowledge.

This experiment measures the improvement of the complete RAG-assisted second-pass workflow.

It does not independently isolate the effect of RAG from the effect of performing a second inference with additional context.

A future optional experiment may introduce a control condition:

- Second Pass A: Original image + First Pass information, without RAG.
- Second Pass B: Original image + First Pass information + RAG knowledge.

This additional experiment can help identify whether improvements come specifically from retrieved knowledge.

It is not required for the initial implementation.
