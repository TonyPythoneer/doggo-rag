import { useEffect, useState } from "react";

import {
  apiErrorMessage,
  fetchDatasetImages,
  imagePreviewSource,
  runPipeline,
  uploadImage,
} from "./api/client";
import { ImagePicker } from "./components/ImagePicker";
import { Mascot, Paw } from "./components/icons";
import { PassCard } from "./components/PassCard";
import { PipelineDiagram } from "./components/PipelineDiagram";
import { StepCard } from "./components/StepCard";
import {
  MODELS,
  type DatasetImage,
  type ModelId,
  type PipelineResponse,
  type StepName,
} from "./types/pipeline";

export default function App() {
  const [modelId, setModelId] = useState<ModelId>("lfm3-vl-2b");
  const [source, setSource] = useState<"local" | "upload">("local");
  const [file, setFile] = useState<File | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [datasetImages, setDatasetImages] = useState<DatasetImage[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [selectedStep, setSelectedStep] = useState<StepName>("first_pass");

  useEffect(() => {
    fetchDatasetImages()
      .then(setDatasetImages)
      .catch((err) => setCatalogError(apiErrorMessage(err, "Could not load dataset images.")))
      .finally(() => setCatalogLoading(false));
  }, []);

  const releasePreview = () => {
    if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
  };

  const pickUpload = (nextFile: File) => {
    releasePreview();
    setFile(nextFile);
    setSelectedImage(null);
    setSelectedName("Uploaded image");
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setError(null);
  };

  const selectDatasetImage = (image: DatasetImage) => {
    releasePreview();
    setFile(null);
    setSelectedImage(`dataset:${image.image_id}`);
    setSelectedName(image.name);
    setPreviewUrl(imagePreviewSource(image.preview_url));
    setResult(null);
    setError(null);
  };

  const clearImage = () => {
    releasePreview();
    setFile(null);
    setSelectedImage(null);
    setSelectedName(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleRun = async () => {
    if (!file && !selectedImage) {
      setError("Select an animal image first.");
      return;
    }
    setStatus(file ? "Uploading image…" : "Running…");
    setError(null);
    setResult(null);
    setSelectedStep("first_pass");
    try {
      let imageReference = selectedImage;
      if (file) {
        const upload = await uploadImage(file);
        imageReference = upload.image;
        setStatus("Running First Pass, retrieval, and Second Pass…");
      }
      if (!imageReference) throw new Error("Select an animal image first.");
      const response = await runPipeline(imageReference, modelId);
      setResult(response);
      if (response.status === "error") {
        setError(response.error || "One or more inference stages failed.");
      }
    } catch (err) {
      setError(apiErrorMessage(err, "Request failed"));
    } finally {
      setStatus("");
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-inner">
          <Mascot className="brand-mark" />
          <div>
            <h1>DOGGO RAG</h1>
            <p>
              Compare how the same vision model classifies an animal before and after it fetches
              animal knowledge.
            </p>
          </div>
        </div>
      </header>

      <main className="app-shell">
        <section className="card setup">
          <ImagePicker
            source={source}
            onSourceChange={setSource}
            images={datasetImages}
            loading={catalogLoading}
            error={catalogError}
            selectedImage={selectedImage}
            disabled={Boolean(status)}
            onSelectImage={selectDatasetImage}
            onUploadFile={pickUpload}
          />

          <div className="setup-panel">
            <h2>Your pick</h2>
            {previewUrl ? (
              <figure className="preview">
                <img src={previewUrl} alt={selectedName || "Selected animal"} />
                <figcaption>
                  <strong>{selectedName}</strong>
                  <button type="button" className="btn btn-quiet" onClick={clearImage}>
                    Remove image
                  </button>
                </figcaption>
              </figure>
            ) : (
              <div className="preview-empty">
                <Mascot className="preview-mascot" />
                No animal picked yet
              </div>
            )}
            <label className="field">
              <span className="field-label">Vision model</span>
              <select
                value={modelId}
                onChange={(event) => setModelId(event.target.value as ModelId)}
              >
                {MODELS.map((model) => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="btn btn-primary btn-block"
              onClick={handleRun}
              disabled={(!file && !selectedImage) || Boolean(status)}
            >
              <Paw />
              Run
            </button>
            {status && (
              <p className="run-status" role="status">
                <Paw className="icon spinner-paw" />
                {status}
              </p>
            )}
            {error && (
              <p className="callout callout-error" role="alert">
                {error}
              </p>
            )}
          </div>
        </section>

        {result && (
          <section className="results" aria-labelledby="results-title">
            <div className="results-header">
              <h2 id="results-title">Results</h2>
              <span className="chip mono">{result.model_id}</span>
              {result.timings.total_s !== undefined && (
                <span className="chip">Total {result.timings.total_s.toFixed(3)} s</span>
              )}
            </div>
            <article className="card pipeline-card">
              <h3>Pipeline</h3>
              <PipelineDiagram
                result={result}
                selectedStep={selectedStep}
                onSelect={setSelectedStep}
              />
            </article>
            {selectedStep === "first_pass" && (
              <PassCard
                step="first_pass"
                index={1}
                title="First Pass"
                result={result.first_pass}
                status={result.step_statuses.first_pass}
                seconds={result.timings.first_pass_s}
                error={result.stage_errors.first_pass}
                rawOutput={result.raw_outputs.first_pass ?? null}
                selected
              />
            )}
            {selectedStep === "retrieval" && (
              <StepCard
                step="retrieval"
                index={2}
                title="Retrieved knowledge"
                status={result.step_statuses.retrieval}
                seconds={result.timings.retrieval_s}
                error={result.stage_errors.retrieval}
                rawOutput={
                  result.retrieval_query === null && !result.retrieved_documents.length
                    ? null
                    : JSON.stringify(
                        { query: result.retrieval_query, documents: result.retrieved_documents },
                        null,
                        2,
                      )
                }
                selected
              >
                <div className="text-block">
                  <h4>Query</h4>
                  <p className="query">{result.retrieval_query || "Not run"}</p>
                </div>
                {result.retrieved_documents.length > 0 ? (
                  <div className="documents">
                    {result.retrieved_documents.map((document) => (
                      <details className="document" key={document.document_id}>
                        <summary>
                          <span className="document-title">{document.title}</span>
                          <span className="chip mono">score {document.score}</span>
                        </summary>
                        <p>{document.content}</p>
                      </details>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No documents retrieved.</p>
                )}
              </StepCard>
            )}
            {selectedStep === "second_pass" && (
              <PassCard
                step="second_pass"
                index={3}
                title="Second Pass"
                result={result.second_pass}
                status={result.step_statuses.second_pass}
                seconds={result.timings.second_pass_s}
                error={result.stage_errors.second_pass}
                rawOutput={result.raw_outputs.second_pass ?? null}
                selected
              />
            )}
          </section>
        )}
      </main>
    </div>
  );
}
