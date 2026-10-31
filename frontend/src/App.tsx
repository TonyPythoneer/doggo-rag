import React, { useState } from "react";
import axios, { AxiosError } from "axios";

interface FirstPassResult {
  is_dog: boolean | null;
  confidence: number;
  description: string;
}

interface SecondPassResult {
  is_dog: boolean | null;
  confidence: number;
}

interface RetrievedDocument {
  document_id: string;
  title: string;
  content: string;
  score: number;
}

interface PipelineResponse {
  image: string;
  model_id: string;
  status: "ok" | "error";
  error: string | null;
  first_pass: FirstPassResult | null;
  retrieval_query: string | null;
  retrieved_documents: RetrievedDocument[];
  retrieved_document_ids: string[];
  second_pass: SecondPassResult | null;
  timings: Record<string, number>;
  stage_errors: Record<string, string>;
  raw_outputs: Partial<Record<"first_pass" | "second_pass", string>>;
  step_statuses: Record<StepName, StepStatus>;
}

interface UploadedImage {
  image: string;
  preview_url: string;
}

interface DatasetImage {
  image_id: string;
  name: string;
  species: string;
  preview_url: string;
}

const API = axios.create({
  baseURL:
    import.meta.env.DEV
      ? `http://${window.location.hostname}:${import.meta.env.VITE_API_PORT ?? "8888"}/api/v1`
      : "/api/v1",
  timeout: Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 300000),
});

const KEY_STORAGE = "doggo-api-key";
const askForKey = () => {
  const key = window.prompt("Enter the API key") ?? "";
  sessionStorage.setItem(KEY_STORAGE, key);
  return key;
};
API.interceptors.request.use((config) => {
  config.headers["X-API-Key"] = sessionStorage.getItem(KEY_STORAGE) ?? askForKey();
  return config;
});
API.interceptors.response.use(undefined, (error) => {
  if (error.response?.status === 401) sessionStorage.removeItem(KEY_STORAGE);
  return Promise.reject(error);
});

const imagePreviewSource = (path: string) =>
  import.meta.env.DEV
    ? `http://${window.location.hostname}:8888${path}`
    : path;

const MODELS = [
  { value: "lfm3-vl-2b", label: "LFM3 VL 2B (Default)" },
  { value: "lfm2-vl-450m", label: "LFM2 VL 450M" },
  { value: "qwen3-vl-2b", label: "Qwen3 VL 2B" },
] as const;

type ModelId = (typeof MODELS)[number]["value"];
type StepName = "first_pass" | "retrieval" | "second_pass";
type StepStatus = "not_started" | "completed" | "failed" | "skipped";

const SPECIES_HEADINGS: Record<string, string> = {
  bear: "Bears",
  cat: "Cats",
  coyote: "Coyotes",
  dingo: "Dingoes",
  dog: "Dogs",
  fox: "Foxes",
  hyena: "Hyenas",
  koala: "Koalas",
  raccoon: "Raccoons",
  red_panda: "Red pandas",
  wolf: "Wolves",
};

function speciesHeading(species: string): string {
  return SPECIES_HEADINGS[species] ?? species.replace(/_/g, " ");
}

// Dogs first, then the other species alphabetically by heading.
function groupBySpecies(images: DatasetImage[]): [string, DatasetImage[]][] {
  const groups = new Map<string, DatasetImage[]>();
  for (const image of images) {
    groups.set(image.species, [...(groups.get(image.species) ?? []), image]);
  }
  return Array.from(groups.entries()).sort(([a], [b]) => {
    if (a === "dog") return -1;
    if (b === "dog") return 1;
    return speciesHeading(a).localeCompare(speciesHeading(b));
  });
}

function dogLabel(isDog: boolean | null): string {
  if (isDog === null) return "Uncertain";
  return isDog ? "Domestic dog" : "Not a domestic dog";
}

// Raw text is kept only when the model produced output; a parse failure leaves none.
function RawOutput({ text }: { text: string | null }) {
  const [search, setSearch] = useState("");
  if (text === null) return <p className="raw-missing">No raw output was captured for this step.</p>;
  const lines = text.split("\n");
  const needle = search.trim().toLowerCase();
  const matches = needle ? lines.filter((line) => line.toLowerCase().includes(needle)) : lines;
  return (
    <details className="raw-output">
      <summary>Raw output</summary>
      <input
        type="search"
        className="raw-search"
        placeholder="Search raw output"
        aria-label="Search raw output"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
      />
      <p className="raw-count">{matches.length} of {lines.length} lines</p>
      <pre>{matches.join("\n")}</pre>
    </details>
  );
}

const STATUS_LABEL: Record<StepStatus, string> = {
  completed: "completed",
  failed: "failed",
  skipped: "skipped",
  not_started: "not started",
};

// Glyphs keep status readable without relying on color alone.
const STATUS_ICON: Record<StepStatus, string> = {
  completed: "✓",
  failed: "✕",
  skipped: "–",
  not_started: "○",
};

const PIPELINE_STEPS = [
  { step: "first_pass", title: "First Pass", detail: "Vision model" },
  { step: "retrieval", title: "RAG Retrieval", detail: "Knowledge search" },
  { step: "second_pass", title: "Second Pass", detail: "Vision model" },
] as const;

function Paw({ className = "icon" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
      <ellipse cx="12" cy="16.5" rx="5.2" ry="4.4" />
      <ellipse cx="5.2" cy="10.4" rx="2.1" ry="2.7" transform="rotate(-22 5.2 10.4)" />
      <ellipse cx="9.4" cy="6.4" rx="2.2" ry="2.8" transform="rotate(-8 9.4 6.4)" />
      <ellipse cx="14.6" cy="6.4" rx="2.2" ry="2.8" transform="rotate(8 14.6 6.4)" />
      <ellipse cx="18.8" cy="10.4" rx="2.1" ry="2.7" transform="rotate(22 18.8 10.4)" />
    </svg>
  );
}

function Mascot({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true">
      <ellipse cx="15" cy="27" rx="8.5" ry="15" transform="rotate(24 15 27)" fill="#8b5a3c" />
      <ellipse cx="49" cy="27" rx="8.5" ry="15" transform="rotate(-24 49 27)" fill="#8b5a3c" />
      <circle cx="32" cy="35" r="19" fill="#f1bf86" />
      <ellipse cx="40" cy="27" rx="6" ry="5" fill="#e3a464" />
      <ellipse cx="32" cy="44" rx="11" ry="8" fill="#fde9cf" />
      <circle cx="25" cy="33" r="2.6" fill="#2e2219" />
      <circle cx="39" cy="33" r="2.6" fill="#2e2219" />
      <circle cx="25.9" cy="32.1" r="0.8" fill="#fff" />
      <circle cx="39.9" cy="32.1" r="0.8" fill="#fff" />
      <ellipse cx="32" cy="40" rx="4" ry="2.9" fill="#2e2219" />
      <path d="M32 42.5v2.5M27.5 45c2.2 2 6.8 2 9 0" stroke="#2e2219" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      <path d="M30 46.5h4v2.6a2 2 0 0 1-4 0z" fill="#f08a8a" />
    </svg>
  );
}

// Rounds to the displayed precision so a 0.0 pp change is not colored as a rise or drop.
function trend(delta: number): "up" | "flat" | "down" {
  const pp = Math.round(delta * 1000) / 10;
  return pp > 0 ? "up" : pp < 0 ? "down" : "flat";
}

function StatusBadge({ status, seconds, confidence, delta }: { status: StepStatus; seconds?: number; confidence?: number; delta?: number }) {
  return (
    <span className="status-line">
      <span className="status-badge" data-status={status}>
        <span aria-hidden="true">{STATUS_ICON[status]}</span> {STATUS_LABEL[status]}
      </span>
      {seconds !== undefined && <span className="status-badge" data-status={status}>{seconds.toFixed(3)} s</span>}
      {confidence !== undefined && <span className="status-badge" data-status={status}>Confidence {(confidence * 100).toFixed(1)}%</span>}
      {delta !== undefined && (
        <span className="status-badge" data-trend={trend(delta)}>
          vs First Pass {delta > 0 ? "+" : ""}{(Math.round(delta * 1000) / 10).toFixed(1)} pp
        </span>
      )}
    </span>
  );
}

interface StepCardProps {
  step: StepName;
  index: number;
  title: string;
  status: StepStatus;
  seconds?: number;
  error?: string;
  rawOutput: string | null;
  selected: boolean;
  children?: React.ReactNode;
}

function StepCard({ step, index, title, status, seconds, error, rawOutput, selected, children }: StepCardProps) {
  return (
    <article id={`step-${step}`} className={`card step-card${selected ? " selected" : ""}`}>
      <header className="step-card-header">
        <span className="step-index" aria-hidden="true">{index}</span>
        <h3>{title}</h3>
        <StatusBadge status={status} seconds={seconds} />
      </header>
      {error && <p className="callout callout-error"><strong>Error:</strong> {error}</p>}
      {children}
      <RawOutput text={rawOutput} />
    </article>
  );
}

function PassCard({
  result,
  ...step
}: Omit<StepCardProps, "children"> & { result: FirstPassResult | SecondPassResult | null }) {
  const confidence = result ? result.confidence * 100 : 0;
  return (
    <StepCard {...step}>
      {result ? (
        <>
          <dl className="pass-result">
            <div>
              <dt>Classification</dt>
              <dd className="classification">{dogLabel(result.is_dog)}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>
                <span className="confidence-value">{confidence.toFixed(1)}%</span>
                <span className="meter" aria-hidden="true">
                  <span style={{ width: `${Math.min(100, Math.max(0, confidence))}%` }} />
                </span>
              </dd>
            </div>
          </dl>
          {"description" in result && (
            <div className="text-block">
              <h4>Visible description</h4>
              <p>{result.description}</p>
            </div>
          )}
        </>
      ) : (
        !step.error && <p className="muted">No result for this step.</p>
      )}
    </StepCard>
  );
}

function PipelineDiagram({
  result,
  selectedStep,
  onSelect,
}: {
  result: PipelineResponse;
  selectedStep: StepName;
  onSelect: (step: StepName) => void;
}) {
  return (
    <ol className="pipeline" aria-label="Pipeline steps">
      <li className="pipeline-item">
        <div className="pipeline-node terminal">
          <span className="node-title">Input</span>
          <span className="node-detail">Animal image</span>
        </div>
      </li>
      {PIPELINE_STEPS.map(({ step, title, detail }, index) => (
        <li className="pipeline-item" key={step}>
          <button
            type="button"
            className="pipeline-node"
            data-status={result.step_statuses[step]}
            aria-pressed={selectedStep === step}
            onClick={() => onSelect(step)}
          >
            <span className="node-text">
              <span className="node-step">Step {index + 1}</span>
              <span className="node-title">{title}</span>
              <span className="node-detail">{detail}</span>
            </span>
            <span className="node-divider" aria-hidden="true" />
            <StatusBadge
              status={result.step_statuses[step]}
              seconds={result.timings[`${step}_s`]}
              confidence={step === "retrieval" ? undefined : result[step]?.confidence}
              delta={step === "second_pass" && result.first_pass && result.second_pass
                ? result.second_pass.confidence - result.first_pass.confidence
                : undefined}
            />
          </button>
        </li>
      ))}
      <li className="pipeline-item">
        <div className="pipeline-node terminal">
          <span className="node-title">Output</span>
          <span className="node-detail node-output">
            {result.second_pass ? dogLabel(result.second_pass.is_dog) : "Not produced"}
          </span>
        </div>
      </li>
    </ol>
  );
}

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

  React.useEffect(() => {
    API.get<DatasetImage[]>("/images")
      .then((response) => setDatasetImages(response.data))
      .catch((err) => {
        const apiError = err as AxiosError<{ detail?: string }>;
        setCatalogError(apiError.response?.data?.detail || "Could not load dataset images.");
      })
      .finally(() => setCatalogLoading(false));
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    event.currentTarget.value = "";
    if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setSelectedImage(null);
    setSelectedName("Uploaded image");
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setError(null);
  };

  const selectDatasetImage = (image: DatasetImage) => {
    if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setSelectedImage(`dataset:${image.image_id}`);
    setSelectedName(image.name);
    setPreviewUrl(imagePreviewSource(image.preview_url));
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
        const form = new FormData();
        form.append("file", file);
        const upload = await API.post<UploadedImage>("/images/uploads", form);
        imageReference = upload.data.image;
        setStatus("Running First Pass, retrieval, and Second Pass…");
      }
      const response = await API.post<PipelineResponse>("/pipeline/predict", {
        image: imageReference,
        model_id: modelId,
      });
      setResult(response.data);
      if (response.data.status === "error") {
        setError(response.data.error || "One or more inference stages failed.");
      }
    } catch (err) {
      const apiError = err as AxiosError<{ detail?: string }>;
      setError(apiError.response?.data?.detail || apiError.message || "Request failed");
    } finally {
      setStatus("");
    }
  };

  const selectStep = (step: StepName) => {
    setSelectedStep(step);
  };

  const clearImage = () => {
    if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setSelectedImage(null);
    setSelectedName(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-inner">
          <Mascot className="brand-mark" />
          <div>
            <h1>DOGGO RAG</h1>
            <p>Compare how the same vision model classifies an animal before and after it fetches animal knowledge.</p>
          </div>
        </div>
      </header>

      <main className="app-shell">
        <section className="card setup">
          <div className="setup-picker">
            <div className="section-heading">
              <h2>Pick an animal</h2>
              <p>Dataset images are grouped by species. The dog / non-dog label is not shown.</p>
            </div>
            <div className="source-switch" role="tablist" aria-label="Image source">
              <button type="button" role="tab" aria-selected={source === "local"} className={source === "local" ? "active" : ""} onClick={() => setSource("local")}>Local pick</button>
              <button type="button" role="tab" aria-selected={source === "upload"} className={source === "upload" ? "active" : ""} onClick={() => setSource("upload")}>Remote upload</button>
            </div>
            <div className="picker-body">
            {source === "local" && catalogLoading && <p className="muted">Loading dataset images…</p>}
            {source === "local" && catalogError && <p className="callout callout-error" role="alert">{catalogError}</p>}
            {source === "local" && !catalogLoading && !catalogError && datasetImages.length === 0 && (
              <p className="muted">No dataset images are available.</p>
            )}
            {source === "local" && datasetImages.length > 0 && (
              <div className="dataset-image-grid" aria-label="Dataset images">
                {groupBySpecies(datasetImages).map(([species, images]) => (
                  <section className="dataset-species-group" key={species}>
                    <h3>{speciesHeading(species)}</h3>
                    <div className="dataset-species-images">
                      {images.map((image) => (
                        <button
                          className={`dataset-image-choice${selectedImage === `dataset:${image.image_id}` ? " selected" : ""}`}
                          key={image.image_id}
                          type="button"
                          disabled={Boolean(status)}
                          aria-label={`Select ${image.name}`}
                          aria-pressed={selectedImage === `dataset:${image.image_id}`}
                          onClick={() => selectDatasetImage(image)}
                        >
                          <img src={imagePreviewSource(image.preview_url)} alt="" loading="lazy" />
                          <span>{image.name}</span>
                          {selectedImage === `dataset:${image.image_id}` && (
                            <span className="paw-badge"><Paw /></span>
                          )}
                        </button>
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            )}
            {source === "upload" && (
              <label className="dropzone">
                <Mascot className="dropzone-mascot" />
                <strong>Click to choose an image</strong>
                <span>JPG, PNG and other image formats</span>
                <input type="file" accept="image/*" onChange={handleFileChange} />
              </label>
            )}
            </div>
          </div>

          <div className="setup-panel">
            <h2>Your pick</h2>
            {previewUrl ? (
              <figure className="preview">
                <img src={previewUrl} alt={selectedName || "Selected animal"} />
                <figcaption>
                  <strong>{selectedName}</strong>
                  <button type="button" className="btn btn-quiet" onClick={clearImage}>Remove image</button>
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
              <select value={modelId} onChange={(event) => setModelId(event.target.value as ModelId)}>
                {MODELS.map((model) => (
                  <option key={model.value} value={model.value}>{model.label}</option>
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
            {error && <p className="callout callout-error" role="alert">{error}</p>}
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
              <PipelineDiagram result={result} selectedStep={selectedStep} onSelect={selectStep} />
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
              selected={selectedStep === "first_pass"}
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
              rawOutput={result.retrieval_query === null && !result.retrieved_documents.length
                ? null
                : JSON.stringify({ query: result.retrieval_query, documents: result.retrieved_documents }, null, 2)}
              selected={selectedStep === "retrieval"}
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
              selected={selectedStep === "second_pass"}
            />
            )}
          </section>
        )}
      </main>
    </div>
  );
}
