import { useState } from "react";

// Raw text is kept only when the model produced output; a parse failure leaves none.
export function RawOutput({ text }: { text: string | null }) {
  const [search, setSearch] = useState("");
  if (text === null)
    return <p className="raw-missing">No raw output was captured for this step.</p>;
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
      <p className="raw-count">
        {matches.length} of {lines.length} lines
      </p>
      <pre>{matches.join("\n")}</pre>
    </details>
  );
}
