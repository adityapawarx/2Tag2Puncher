"use client";

import type { Tag } from "@/app/api";

type Props = {
  tag: Tag;
  max: number;
  active: boolean;
  onHover: (tag: Tag | null) => void;
};

export function TagChip({ tag, max, active, onHover }: Props) {
  const width = max > 0 ? Math.max(4, (tag.score / max) * 100) : 0;

  return (
    <button
      type="button"
      onMouseEnter={() => onHover(tag)}
      onMouseLeave={() => onHover(null)}
      className={`relative overflow-hidden rounded-full border px-3 py-1 text-sm transition ${
        active ? "border-neutral-900 bg-white" : "border-neutral-300 bg-white hover:border-neutral-500"
      }`}
      title={`score ${tag.score.toFixed(3)} — from chunk${tag.chunks.length > 1 ? "s" : ""} ${tag.chunks.join(", ")}`}
    >
      <span
        aria-hidden
        className="absolute inset-y-0 left-0 bg-neutral-900/10"
        style={{ width: `${width}%` }}
      />
      <span className="relative">{tag.tag.replace(/_/g, " ")}</span>
      <span className="relative ml-2 text-xs tabular-nums text-neutral-500">
        {tag.score.toFixed(2)}
      </span>
    </button>
  );
}
