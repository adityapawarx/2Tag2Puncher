"use client";

import { useMemo, useState } from "react";
import { TagChip } from "@/components/TagChip";
import { tagFile, tagText, type Tag, type TagResponse } from "./api";

const SAMPLE = `We introduce a sparse retrieval method for extreme multi-label classification.
Documents are represented with TF-IDF features and labels are organised into a
hierarchical tree, which reduces inference from linear in the number of labels to
logarithmic while preserving precision at the top of the ranking.`;

export default function Home() {
  const [text, setText] = useState(SAMPLE);
  const [topK, setTopK] = useState(10);
  const [result, setResult] = useState<TagResponse | null>(null);
  const [hovered, setHovered] = useState<Tag | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const maxScore = useMemo(
    () => result?.tags.reduce((best, tag) => Math.max(best, tag.score), 0) ?? 0,
    [result],
  );

  async function run(action: () => Promise<TagResponse>) {
    setBusy(true);
    setError(null);
    try {
      setResult(await action());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "request failed");
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-6 px-6 py-12">
      <header>
        <h1 className="text-2xl font-semibold">TagPuncher</h1>
        <p className="mt-1 text-sm text-neutral-600">
          Concept tags for any text, learned from the links Wikipedia editors already wrote.
        </p>
      </header>

      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        rows={10}
        aria-label="Text to tag"
        className="w-full resize-y rounded-lg border border-neutral-300 bg-white p-4 font-mono text-sm outline-none focus:border-neutral-900"
      />

      <div className="flex flex-wrap items-center gap-4">
        <button
          type="button"
          disabled={busy || text.trim().length === 0}
          onClick={() => run(() => tagText(text, topK))}
          className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy ? "Tagging…" : "Tag text"}
        </button>

        <label className="flex items-center gap-2 text-sm text-neutral-600">
          <span>Tags</span>
          <input
            type="range"
            min={1}
            max={30}
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
            className="w-40"
          />
          <span className="w-6 tabular-nums">{topK}</span>
        </label>

        <label className="cursor-pointer text-sm text-neutral-600 underline">
          <input
            type="file"
            accept=".pdf,.txt,.md"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                run(() => tagFile(file, topK));
              }
            }}
          />
          or upload a paper
        </label>
      </div>

      {error && (
        <p role="alert" className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </p>
      )}

      {result && (
        <section className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-2">
            {result.tags.length === 0 && (
              <p className="text-sm text-neutral-500">No tags above the score threshold.</p>
            )}
            {result.tags.map((tag) => (
              <TagChip
                key={tag.tag}
                tag={tag}
                max={maxScore}
                active={hovered?.tag === tag.tag}
                onHover={setHovered}
              />
            ))}
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-xs uppercase tracking-wide text-neutral-500">
              {result.chunks.length} chunk{result.chunks.length === 1 ? "" : "s"} · model{" "}
              {result.model_version}
              {hovered && " · highlighted chunks produced the hovered tag"}
            </p>
            {result.chunks.map((chunk, index) => (
              <p
                key={index}
                className={`rounded-md border p-3 text-sm transition ${
                  hovered?.chunks.includes(index)
                    ? "border-neutral-900 bg-white"
                    : "border-neutral-200 bg-neutral-100 text-neutral-500"
                }`}
              >
                {chunk.slice(0, 400)}
                {chunk.length > 400 && "…"}
              </p>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
