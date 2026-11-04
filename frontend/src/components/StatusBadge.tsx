import type { StepStatus } from "../types/pipeline";

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

// Rounds to the displayed precision so a 0.0 pp change is not colored as a rise or drop.
function trend(delta: number): "up" | "flat" | "down" {
  const pp = Math.round(delta * 1000) / 10;
  return pp > 0 ? "up" : pp < 0 ? "down" : "flat";
}

interface StatusBadgeProps {
  status: StepStatus;
  seconds?: number;
  confidence?: number;
  delta?: number;
}

export function StatusBadge({ status, seconds, confidence, delta }: StatusBadgeProps) {
  return (
    <span className="status-line">
      <span className="status-badge" data-status={status}>
        <span aria-hidden="true">{STATUS_ICON[status]}</span> {STATUS_LABEL[status]}
      </span>
      {seconds !== undefined && (
        <span className="status-badge" data-status={status}>
          {seconds.toFixed(3)} s
        </span>
      )}
      {confidence !== undefined && (
        <span className="status-badge" data-status={status}>
          Confidence {(confidence * 100).toFixed(1)}%
        </span>
      )}
      {delta !== undefined && (
        <span className="status-badge" data-trend={trend(delta)}>
          vs First Pass {delta > 0 ? "+" : ""}
          {(Math.round(delta * 1000) / 10).toFixed(1)} pp
        </span>
      )}
    </span>
  );
}
