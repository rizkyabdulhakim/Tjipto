import type { Citation } from "./types";

const API_BASE =
  import.meta.env.VITE_TJIPTO_API_BASE ??
  "http://localhost:8000";

export interface ValidationReasons {
  [evidenceId: string]: string;
}

export interface TjiptoAskResponse {
  status: string;
  answer_type?: string;
  answer?: string;
  route?: string;
  intent?: string;
  evidence?: EvidencePayload[];
  citations?: CitationPayload[];
  viewer_refs?: ViewerRefPayload[];
  context_pack?: {
    answer_evidence?: EvidencePayload[];
    supporting_context?: EvidencePayload[];
    excluded_results?: EvidencePayload[];
    citation_payloads?: CitationPayload[];
    viewer_refs?: ViewerRefPayload[];
    validation_reasons?: ValidationReasons;
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

export interface ViewerRefPayload {
  action?: string;
  evidence_id: string;
  page_numbers?: number[];
  bbox_count?: number;
  source_pdf_path?: string;
  source_sha256?: string;
  can_resolve?: boolean;
}

export interface CitationPayload {
  corpus_id?: string;
  evidence_id: string;
  legal_unit_id?: string;
  source_document_id?: string;
  citation?: string;
  label?: string;
  hierarchy?: string[];
  quoted_text: string;
  source_role?: string;
  temporal_context?: string;
  source_pdf_path?: string;
  source_sha256?: string;
  page_numbers?: number[];
  bbox_count?: number;
  viewer_ref?: ViewerRefPayload;
  evidence_status?: string;
}

export interface EvidencePayload extends CitationPayload {
  route_sources?: string[];
}

export interface ViewerPayload {
  status: string;
  corpus_id?: string;
  evidence_id?: string;
  legal_unit_id?: string;
  source_document_id?: string;
  citation?: string;
  quoted_text?: string;
  source_pdf_path?: string;
  source_sha256?: string;
  page_numbers?: number[];
  bbox_count?: number;
  bbox_rectangles?: Array<{
    bbox_id?: string;
    page_number?: number;
    x0?: number;
    y0?: number;
    x1?: number;
    y1?: number;
    width?: number;
    height?: number;
  }>;
  rendering_available?: boolean;
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

export async function getViewerPayload(evidenceId: string): Promise<ViewerPayload> {
  const response = await fetch(`${API_BASE}/uud/viewer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evidence_id: evidenceId }),
  });
  if (!response.ok) throw new Error(`UUD viewer returned ${response.status}`);
  return response.json();
}

export function fallbackAnswer() {
  return "Bukti tidak cukup / database belum tersedia dalam korpus UUD terverifikasi saat ini.";
}

export function mapAskResponseToCitations(response: TjiptoAskResponse): Citation[] {
  const citations = Array.isArray(response.citations) ? response.citations : [];
  const viewerRefs = Array.isArray(response.viewer_refs) ? response.viewer_refs : [];
  return citations.flatMap((item, index) => {
    if (!item?.evidence_id || !item?.quoted_text) return [];
    const viewer = viewerRefs[index] ?? item.viewer_ref;
    const pages = Array.isArray(item.page_numbers)
      ? item.page_numbers
      : Array.isArray(viewer.page_numbers)
        ? viewer.page_numbers
        : [];
    const pageNumber = Number(pages[0] ?? 1);
    return {
      id: index + 1,
      documentId: String(item.evidence_id),
      legalUnitId: item.legal_unit_id,
      sourceDocumentId: item.source_document_id,
      viewerRefId: viewer.evidence_id,
      documentTitle: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
      regulationType: "UUD",
      article: String(item.label ?? item.citation ?? "UUD"),
      pageNumber: Number.isFinite(pageNumber) ? pageNumber : 1,
      excerpt: String(item.quoted_text),
      sourceUrl: "",
      sourceDomain: "UUD runtime",
      fileHash: item.source_sha256 ? `sha256:${String(item.source_sha256)}` : undefined,
    };
  });
}
