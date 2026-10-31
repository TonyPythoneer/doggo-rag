import type { ChangeEvent } from "react";

import { imagePreviewSource } from "../api/client";
import { groupBySpecies, speciesHeading } from "../display";
import type { DatasetImage } from "../types/pipeline";
import { Mascot, Paw } from "./icons";

interface ImagePickerProps {
  source: "local" | "upload";
  onSourceChange: (source: "local" | "upload") => void;
  images: DatasetImage[];
  loading: boolean;
  error: string | null;
  selectedImage: string | null;
  disabled: boolean;
  onSelectImage: (image: DatasetImage) => void;
  onUploadFile: (file: File) => void;
}

export function ImagePicker({
  source,
  onSourceChange,
  images,
  loading,
  error,
  selectedImage,
  disabled,
  onSelectImage,
  onUploadFile,
}: ImagePickerProps) {
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    event.currentTarget.value = "";
    onUploadFile(nextFile);
  };

  return (
    <div className="setup-picker">
      <div className="section-heading">
        <h2>Pick an animal</h2>
        <p>Dataset images are grouped by species. The dog / non-dog label is not shown.</p>
      </div>
      <div className="source-switch" role="tablist" aria-label="Image source">
        <button
          type="button"
          role="tab"
          aria-selected={source === "local"}
          className={source === "local" ? "active" : ""}
          onClick={() => onSourceChange("local")}
        >
          Local pick
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={source === "upload"}
          className={source === "upload" ? "active" : ""}
          onClick={() => onSourceChange("upload")}
        >
          Remote upload
        </button>
      </div>
      <div className="picker-body">
        {source === "local" && loading && <p className="muted">Loading dataset images…</p>}
        {source === "local" && error && (
          <p className="callout callout-error" role="alert">
            {error}
          </p>
        )}
        {source === "local" && !loading && !error && images.length === 0 && (
          <p className="muted">No dataset images are available.</p>
        )}
        {source === "local" && images.length > 0 && (
          <div className="dataset-image-grid" aria-label="Dataset images">
            {groupBySpecies(images).map(([species, speciesImages]) => (
              <section className="dataset-species-group" key={species}>
                <h3>{speciesHeading(species)}</h3>
                <div className="dataset-species-images">
                  {speciesImages.map((image) => {
                    const reference = `dataset:${image.image_id}`;
                    const isSelected = selectedImage === reference;
                    return (
                      <button
                        className={`dataset-image-choice${isSelected ? " selected" : ""}`}
                        key={image.image_id}
                        type="button"
                        disabled={disabled}
                        aria-label={`Select ${image.name}`}
                        aria-pressed={isSelected}
                        onClick={() => onSelectImage(image)}
                      >
                        <img src={imagePreviewSource(image.preview_url)} alt="" loading="lazy" />
                        <span>{image.name}</span>
                        {isSelected && (
                          <span className="paw-badge">
                            <Paw />
                          </span>
                        )}
                      </button>
                    );
                  })}
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
  );
}
