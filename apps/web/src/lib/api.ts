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

export interface SearchResult {
  corpus_id: string;
  legal_unit_id: string;
  evidence_id: string;
  citation_id?: string;
  viewer_ref_id?: string;
  title?: string;
  snippet?: string;
  retrieval_method?: string;
  reasons?: string;
  status: string;
}

export interface BookmarkPointer {
  bookmark_id: string;
  corpus_id: string;
  legal_unit_id: string;
  evidence_id: string;
  citation_id?: string;
  viewer_ref_id?: string;
  note?: string;
  created_at: string;
  status: string;
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

export async function searchUud(query: string): Promise<SearchResult[]> {
  const response = await fetch(`${API_BASE}/uud/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit: 5 }),
  });
  if (!response.ok) throw new Error(`UUD search returned ${response.status}`);
  const body = await response.json();
  return Array.isArray(body.results) ? body.results : [];
}

export async function listBookmarks(): Promise<BookmarkPointer[]> {
  const response = await fetch(`${API_BASE}/uud/bookmarks`);
  if (!response.ok) throw new Error(`UUD bookmarks returned ${response.status}`);
  const body = await response.json();
  return Array.isArray(body.bookmarks) ? body.bookmarks : [];
}

export async function saveBookmark(evidenceId: string): Promise<BookmarkPointer | null> {
  const response = await fetch(`${API_BASE}/uud/bookmarks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evidence_id: evidenceId }),
  });
  if (!response.ok) throw new Error(`UUD bookmark returned ${response.status}`);
  const body = await response.json();
  return body.bookmark ?? null;
}

export function fallbackAnswer() {
  return "Bukti tidak cukup / database belum tersedia dalam korpus UUD terverifikasi saat ini.";
}

export function mapAskResponseToCitations(response: TjiptoAskResponse): Citation[] {
  const citations = Array.isArray(response.citations) ? response.citations : [];
  const viewerRefs = Array.isArray(response.viewer_refs) ? response.viewer_refs : [];
  return citations.flatMap((item, index) => {
    if (!item?.evidence_id || !item?.quoted_text) return [];
    const viewer = viewerRefs[index] ?? item.viewer_ref ?? {};
    const pages = Array.isArray(item.page_numbers)
      ? item.page_numbers
      : Array.isArray(viewer.page_numbers)
        ? viewer.page_numbers
        : [];
    const pageNumber = Number(pages[0] ?? 1);
    return {
      id: index + 1,
      documentId: String(item.evidence_id),
      documentTitle: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
      regulationType: "UUD",
      article: String(item.citation ?? "UUD"),
      pageNumber: Number.isFinite(pageNumber) ? pageNumber : 1,
      excerpt: String(item.quoted_text),
      sourceUrl: "",
      sourceDomain: "UUD runtime",
      fileHash: item.source_sha256 ? `sha256:${String(item.source_sha256)}` : undefined,
    };
  });
}
