import { dogLabel } from "../display";
import type { FirstPassResult, SecondPassResult } from "../types/pipeline";
import { StepCard, type StepCardProps } from "./StepCard";

export function PassCard({
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
