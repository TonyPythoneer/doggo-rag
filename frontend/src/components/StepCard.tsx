import type { ReactNode } from "react";

import type { StepName, StepStatus } from "../types/pipeline";
import { RawOutput } from "./RawOutput";
import { StatusBadge } from "./StatusBadge";

export interface StepCardProps {
  step: StepName;
  index: number;
  title: string;
  status: StepStatus;
  seconds?: number;
  error?: string;
  rawOutput: string | null;
  selected: boolean;
  children?: ReactNode;
}

export function StepCard({
  step,
  index,
  title,
  status,
  seconds,
  error,
  rawOutput,
  selected,
  children,
}: StepCardProps) {
  return (
    <article id={`step-${step}`} className={`card step-card${selected ? " selected" : ""}`}>
      <header className="step-card-header">
        <span className="step-index" aria-hidden="true">
          {index}
        </span>
        <h3>{title}</h3>
        <StatusBadge status={status} seconds={seconds} />
      </header>
      {error && (
        <p className="callout callout-error">
          <strong>Error:</strong> {error}
        </p>
      )}
      {children}
      <RawOutput text={rawOutput} />
    </article>
  );
}
