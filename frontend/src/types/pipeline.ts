/** Shapes returned by the FastAPI backend (app/pipeline/schemas.py). */

export type StepName = "first_pass" | "retrieval" | "second_pass";
export type StepStatus = "not_started" | "completed" | "failed" | "skipped";

export interface FirstPassResult {
  is_dog: boolean | null;
  confidence: number;
  description: string;
}

export interface SecondPassResult {
  is_dog: boolean | null;
  confidence: number;
}

export interface RetrievedDocument {
  document_id: string;
  title: string;
  content: string;
  score: number;
}

export interface PipelineResponse {
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

export interface UploadedImage {
  image: string;
  preview_url: string;
}

export interface DatasetImage {
  image_id: string;
  name: string;
  species: string;
  preview_url: string;
}

// Display names only: the supported IDs are defined once, in app/config.py.
export const MODELS = [
  { value: "lfm3-vl-2b", label: "LFM3 VL 2B (Default)" },
  { value: "lfm2-vl-450m", label: "LFM2 VL 450M" },
  { value: "qwen3-vl-2b", label: "Qwen3 VL 2B" },
] as const;

export type ModelId = (typeof MODELS)[number]["value"];
