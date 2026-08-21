export type Tag = {
  tag: string;
  score: number;
  chunks: number[];
};

export type TagResponse = {
  tags: Tag[];
  chunks: string[];
  model_version: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** FastAPI errors arrive as {"detail": ...}; show the message, not the envelope. */
async function detailOf(response: Response): Promise<string> {
  const body = await response.text();
  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => (item as { msg?: string }).msg ?? "invalid input").join("; ");
    }
  } catch {
    // Not JSON (e.g. a proxy error page) — fall through to the raw body.
  }
  return body;
}

async function parse(response: Response): Promise<TagResponse> {
  if (!response.ok) {
    throw new Error((await detailOf(response)) || `request failed with ${response.status}`);
  }
  return (await response.json()) as TagResponse;
}

export async function tagText(text: string, topK: number): Promise<TagResponse> {
  return parse(
    await fetch(`${API_URL}/v1/tag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k: topK }),
    }),
  );
}

export async function tagFile(file: File, topK: number): Promise<TagResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("top_k", String(topK));
  return parse(await fetch(`${API_URL}/v1/tag/document`, { method: "POST", body: form }));
}
