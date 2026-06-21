import type { Citation } from "./types";

const API_BASE =
  import.meta.env.VITE_TJIPTO_API_BASE ??
  "http://localhost:8000";

export interface TjiptoAskResponse {
  status: string;
  answer_type?: string;
  answer?: string;
  route?: string;
  intent?: string;
  evidence?: Array<Record<string, any>>;
  citations?: Array<Record<string, any>>;
  viewer_refs?: Array<Record<string, any>>;
  context_pack?: {
    answer_evidence?: Array<Record<string, any>>;
    supporting_context?: Array<Record<string, any>>;
    excluded_results?: Array<Record<string, any>>;
    citation_payloads?: Array<Record<string, any>>;
    viewer_refs?: Array<Record<string, any>>;
    validation_reasons?: Record<string, string>;
  };
}

export async function askUud(query: string): Promise<TjiptoAskResponse> {
  const response = await fetch(`${API_BASE}/uud/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) throw new Error(`UUD runtime returned ${response.status}`);
  return response.json();
}

export function fallbackAnswer() {
  return "Bukti tidak cukup / database belum tersedia dalam korpus UUD terverifikasi saat ini.";
}

export function mapAskResponseToCitations(response: TjiptoAskResponse): Citation[] {
  return (response.citations ?? []).map((item, index) => {
    const viewer = response.viewer_refs?.[index] ?? item.viewer_ref ?? {};
    const pages = item.page_numbers ?? viewer.page_numbers ?? [];
    return {
      id: index + 1,
      documentId: String(item.evidence_id ?? `uud-${index + 1}`),
      documentTitle: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
      regulationType: "UUD",
      article: String(item.citation ?? "UUD"),
      pageNumber: Number(pages[0] ?? 1),
      excerpt: String(item.quoted_text ?? ""),
      sourceUrl: "",
      sourceDomain: "UUD runtime",
      fileHash: item.source_sha256 ? `sha256:${String(item.source_sha256)}` : undefined,
    };
  });
}
