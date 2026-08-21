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

async function parse(response: Response): Promise<TagResponse> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `request failed with ${response.status}`);
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
