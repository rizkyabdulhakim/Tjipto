import type { Citation } from "./types";

const API_BASE =
  import.meta.env.VITE_TJIPTO_API_BASE ??
  "http://localhost:8000";
const DEFAULT_CORPUS_ID = import.meta.env.VITE_TJIPTO_CORPUS_ID ?? "uud";

function corpusEndpoint(action: string) {
  return `${API_BASE}/legal/${DEFAULT_CORPUS_ID}/${action}`;
}

export interface ValidationReasons {
  [evidenceId: string]: string;
}

export interface TjiptoAskResponse {
  status: string;
  answer_type?: string;
  answer?: string;
  citations?: CitationPayload[];
  viewer_refs?: ViewerRefPayload[];
}

export interface SearchResult {
  corpus_id: string;
  legal_unit_id: string;
  evidence_id: string;
  citation_id?: string;
  viewer_ref_id?: string;
  source_document_id?: string;
  title?: string;
  document_title?: string;
  citation?: string;
  label?: string;
  snippet?: string;
  source_role?: string;
  temporal_context?: string;
  page_numbers?: number[];
  bbox_count?: number;
  viewer_ref?: ViewerRefPayload;
  status: string;
}

export interface ViewerRefPayload {
  action?: string;
  evidence_id: string;
  page_numbers?: number[];
  bbox_count?: number;
  source_role?: string;
  temporal_context?: string;
  source_status_label?: string;
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
  document_title?: string;
  quoted_text: string;
  source_role?: string;
  temporal_context?: string;
  page_numbers?: number[];
  bbox_count?: number;
  viewer_ref?: ViewerRefPayload;
  evidence_status?: string;
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
  source_role?: string;
  temporal_context?: string;
  source_status_label?: string;
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
    x?: number;
    y?: number;
    page_width?: number;
    page_height?: number;
  }>;
  pdf_access_available?: boolean;
  rendering_available?: boolean;
  render_status?: string;
  page_number?: number;
  page_width?: number;
  page_height?: number;
  reason?: string | null;
  render?: {
    mime_type?: string;
    width?: number;
    height?: number;
  };
  pdf?: {
    mime_type?: string;
    access_url?: string;
  };
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

export interface BookmarksResponse {
  persistence?: "memory" | string;
  persistence_label?: string;
  bookmarks: BookmarkPointer[];
}

export async function askLegal(query: string): Promise<TjiptoAskResponse> {
  const response = await fetch(corpusEndpoint("ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) throw new Error(`${DEFAULT_CORPUS_ID} runtime returned ${response.status}`);
  return response.json();
}

export async function searchLegal(query: string): Promise<SearchResult[]> {
  const response = await fetch(corpusEndpoint("search"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit: 5 }),
  });
  if (!response.ok) throw new Error(`${DEFAULT_CORPUS_ID} search returned ${response.status}`);
  const body = await response.json();
  return Array.isArray(body.results) ? body.results : [];
}

export async function listLegalBookmarks(): Promise<BookmarksResponse> {
  const response = await fetch(corpusEndpoint("bookmarks"));
  if (!response.ok) throw new Error(`${DEFAULT_CORPUS_ID} bookmarks returned ${response.status}`);
  const body = await response.json();
  return {
    persistence: body.persistence,
    persistence_label: body.persistence_label,
    bookmarks: Array.isArray(body.bookmarks) ? body.bookmarks : [],
  };
}

export async function saveLegalBookmark(evidenceId: string): Promise<BookmarkPointer | null> {
  const response = await fetch(corpusEndpoint("bookmarks"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evidence_id: evidenceId }),
  });
  if (!response.ok) throw new Error(`${DEFAULT_CORPUS_ID} bookmark returned ${response.status}`);
  const body = await response.json();
  return body.bookmark ?? null;
}

export async function getLegalViewerPayload(evidenceId: string): Promise<ViewerPayload> {
  const response = await fetch(corpusEndpoint("viewer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evidence_id: evidenceId }),
  });
  if (!response.ok) throw new Error(`${DEFAULT_CORPUS_ID} viewer returned ${response.status}`);
  return response.json();
}

export function pdfAccessUrl(viewer: ViewerPayload): string | null {
  const accessUrl = viewer.pdf?.access_url;
  if (!accessUrl) return null;
  return new URL(accessUrl, API_BASE).toString();
}

export const askUud = askLegal;
export const searchUud = searchLegal;
export const listBookmarks = listLegalBookmarks;
export const saveBookmark = saveLegalBookmark;
export const getViewerPayload = getLegalViewerPayload;

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
      : Array.isArray(viewer?.page_numbers)
        ? viewer.page_numbers
        : [];
    const pageNumber = Number(pages[0] ?? 1);
    return {
      id: index + 1,
      documentId: String(item.evidence_id),
      legalUnitId: item.legal_unit_id,
      sourceDocumentId: item.source_document_id,
      viewerRefId: viewer?.evidence_id,
      documentTitle: item.document_title ?? fallbackDocumentTitle(item.corpus_id),
      regulationType: "UUD",
      article: String(item.label ?? item.citation ?? "UUD"),
      pageNumber: Number.isFinite(pageNumber) ? pageNumber : 1,
      excerpt: String(item.quoted_text),
      sourceUrl: "",
      sourceDomain: item.source_role ?? item.corpus_id ?? "runtime",
      sourceRole: item.source_role,
      temporalContext: item.temporal_context,
      sourceStatusLabel: sourceStatusLabel(item.source_role, item.temporal_context),
    };
  });
}

export function mapSearchResultToCitation(item: SearchResult, index: number): Citation | null {
  if (!item?.evidence_id || !item.snippet) return null;
  const pages = Array.isArray(item.page_numbers)
    ? item.page_numbers
    : Array.isArray(item.viewer_ref?.page_numbers)
      ? item.viewer_ref.page_numbers
      : [];
  const pageNumber = Number(pages[0] ?? 1);
  return {
    id: index + 1,
    documentId: String(item.evidence_id),
    legalUnitId: item.legal_unit_id,
    sourceDocumentId: item.source_document_id,
    viewerRefId: item.viewer_ref_id ?? item.viewer_ref?.evidence_id,
    documentTitle: item.document_title ?? fallbackDocumentTitle(item.corpus_id),
    regulationType: "UUD",
    article: String(item.label ?? item.citation ?? item.title ?? "UUD"),
    pageNumber: Number.isFinite(pageNumber) ? pageNumber : 1,
    excerpt: String(item.snippet),
    sourceUrl: "",
    sourceDomain: item.source_role ?? item.corpus_id ?? "runtime",
    sourceRole: item.source_role,
    temporalContext: item.temporal_context,
    sourceStatusLabel: sourceStatusLabel(item.source_role, item.temporal_context),
  };
}

function fallbackDocumentTitle(corpusId?: string) {
  return corpusId === "uud"
    ? "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945"
    : corpusId ?? "Dokumen hukum";
}

function sourceStatusLabel(sourceRole?: string, temporalContext?: string) {
  const role = sourceRole ?? temporalContext;
  if (role === "current_consolidated") return "Berlaku (konsolidasi saat ini)";
  if (role?.startsWith("amendment_")) return "Historis (sumber perubahan)";
  if (role === "original_historical") return "Historis (naskah asli)";
  return undefined;
}
