import axios, { AxiosError } from "axios";

import type { DatasetImage, ModelId, PipelineResponse, UploadedImage } from "../types/pipeline";

// One origin for requests and image previews, so a custom VITE_API_PORT applies
// to both. A production build is served by FastAPI on the same origin.
const API_ORIGIN = import.meta.env.DEV
  ? `http://${window.location.hostname}:${import.meta.env.VITE_API_PORT ?? "8888"}`
  : "";

/** Turn a backend preview path into a URL that this page can load. */
export const imagePreviewSource = (path: string) => `${API_ORIGIN}${path}`;

const KEY_STORAGE = "doggo-api-key";

const askForKey = () => {
  const key = window.prompt("Enter the API key") ?? "";
  sessionStorage.setItem(KEY_STORAGE, key);
  return key;
};

const API = axios.create({
  baseURL: `${API_ORIGIN}/api/v1`,
  timeout: Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 300000),
});

API.interceptors.request.use((config) => {
  config.headers["X-API-Key"] = sessionStorage.getItem(KEY_STORAGE) ?? askForKey();
  return config;
});
API.interceptors.response.use(undefined, (error) => {
  if (error.response?.status === 401) sessionStorage.removeItem(KEY_STORAGE);
  return Promise.reject(error);
});

/** The backend's `detail` message, or a readable fallback. */
export function apiErrorMessage(error: unknown, fallback: string): string {
  const apiError = error as AxiosError<{ detail?: string }>;
  return apiError.response?.data?.detail || apiError.message || fallback;
}

export async function fetchDatasetImages(): Promise<DatasetImage[]> {
  const response = await API.get<DatasetImage[]>("/images");
  return response.data;
}

export async function uploadImage(file: File): Promise<UploadedImage> {
  const form = new FormData();
  form.append("file", file);
  const response = await API.post<UploadedImage>("/images/uploads", form);
  return response.data;
}

export async function runPipeline(image: string, modelId: ModelId): Promise<PipelineResponse> {
  const response = await API.post<PipelineResponse>("/pipeline/predict", {
    image,
    model_id: modelId,
  });
  return response.data;
}
