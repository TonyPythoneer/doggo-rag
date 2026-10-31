import { dogLabel } from "../display";
import type { PipelineResponse, StepName } from "../types/pipeline";
import { StatusBadge } from "./StatusBadge";

const PIPELINE_STEPS = [
  { step: "first_pass", title: "First Pass", detail: "Vision model" },
  { step: "retrieval", title: "RAG Retrieval", detail: "Knowledge search" },
  { step: "second_pass", title: "Second Pass", detail: "Vision model" },
] as const;

function passConfidence(result: PipelineResponse, step: StepName): number | undefined {
  if (step === "first_pass") return result.first_pass?.confidence;
  if (step === "second_pass") return result.second_pass?.confidence;
  return undefined;
}

export function PipelineDiagram({
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
              confidence={passConfidence(result, step)}
              delta={
                step === "second_pass" && result.first_pass && result.second_pass
                  ? result.second_pass.confidence - result.first_pass.confidence
                  : undefined
              }
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
