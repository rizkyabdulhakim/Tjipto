import type { Citation, LayoutLine, SupportItem } from "./types";
import type { PdfBBox } from "./pdfBBox";

const API_BASE =
  (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TJIPTO_API_BASE ??
  "http://localhost:8000";
const DEFAULT_CORPUS_ID = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TJIPTO_CORPUS_ID ?? "uud";

function corpusEndpoint(action: string) {
  return `${API_BASE}/legal/${DEFAULT_CORPUS_ID}/${action}`;
}

export interface ValidationReasons {
  [evidenceId: string]: string;
}

export interface TjiptoAskResponse {
  status: string;
  route?: string;
  answer_type?: string;
  answer?: string;
  answer_scope?: string;
  document_source?: DocumentSourcePayload;
  warnings?: string[];
  insufficient_reasons?: string[];
  clarification_options?: ClarificationOptionPayload[];
  supports?: SupportPayload[];
}

export interface SupportPayload {
  support_id?: string;
  support_kind?: string;
  panel_section?: string;
  fact_kind?: string;
  display_label?: string;
  source_document?: string;
  source_role?: string;
  display_text?: string;
  layout_lines?: LayoutLine[];
  copy_text?: string;
  legal_citation_available?: boolean;
  relevant_quote_eligible?: boolean;
  linkable?: boolean;
  viewer_target?: ViewerRefPayload;
  page_numbers?: number[];
  highlightable?: boolean;
}

export interface DocumentSourcePayload {
  source_role?: string;
  temporal_context?: string;
  document_title?: string;
  viewer_target?: {
    action?: "open_document";
    target?: string;
  };
}

export interface ClarificationOptionPayload {
  source_role?: string;
  label?: string;
}

export interface SearchResult {
  corpus_id: string;
  title?: string;
  document_title?: string;
  citation?: string;
  label?: string;
  snippet?: string;
  source_role?: string;
  temporal_context?: string;
  page_numbers?: number[];
  bbox_count?: number;
  viewer_target?: ViewerRefPayload;
  status: string;
}

export interface ViewerRefPayload {
  action?: string;
  target?: string;
  page_numbers?: number[];
  bbox_count?: number;
  source_status_label?: string;
  can_resolve?: boolean;
  source_proof_text_span_ids?: string[];
  source_proof_bbox_refs?: string[];
  target_text_span_ids?: string[];
  target_bbox_refs?: string[];
}

export interface CitationPayload {
  corpus_id?: string;
  evidence_id: string;
  legal_unit_id?: string;
  source_document_id?: string;
  source_url?: string;
  citation?: string;
  label?: string;
  hierarchy?: string[];
  document_title?: string;
  quoted_text: string;
  source_role?: string;
  temporal_context?: string;
  authority_kind?: Citation["authorityKind"];
  authority_label?: string;
  citation_final?: boolean;
  page_numbers?: number[];
  bbox_count?: number;
  viewer_ref?: ViewerRefPayload;
  evidence_status?: string;
  metadata_answer?: string;
  metadata_field?: string;
  support_kind?: string;
  relevant_quote_eligible?: boolean;
  display_text?: string;
  copy_text?: string;
  layout_lines?: LayoutLine[];
  viewer_target?: Record<string, unknown>;
}

export interface ViewerPayload {
  status: string;
  corpus_id?: string;
  evidence_id?: string;
  legal_unit_id?: string;
  source_document_id?: string;
  source_url?: string;
  citation?: string;
  quoted_text?: string;
  source_role?: string;
  temporal_context?: string;
  source_status_label?: string;
  authority_kind?: Citation["authorityKind"];
  authority_label?: string;
  citation_final?: boolean;
  page_numbers?: number[];
  bbox_count?: number;
  bbox_rectangles?: (PdfBBox & {
    bbox_precision?: "exact" | "coarse" | "page_grounded_only";
    viewer_highlightable?: boolean;
    coordinate_space?: string;
    coordinate_origin?: string;
    page_width?: number;
    page_height?: number;
    page_rotation?: number;
    page_box_basis?: string;
    transform_version?: string;
  })[];
  pdf_access_available?: boolean;
  rendering_available?: boolean;
  render_status?: string;
  page_number?: number;
  page_width?: number;
  page_height?: number;
  reason?: string | null;
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

export async function askLegal(query: string, filters?: { source_role: string }): Promise<TjiptoAskResponse> {
  const response = await fetch(corpusEndpoint("ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ...(filters ? { filters } : {}) }),
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

export async function getLegalViewerPayload(evidenceId: string, relationId?: string): Promise<ViewerPayload> {
  const response = await fetch(corpusEndpoint("viewer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: evidenceId }),
  });
  if (!response.ok) throw new Error(`${DEFAULT_CORPUS_ID} viewer returned ${response.status}`);
  return response.json();
}

export async function getDocumentViewerPayload(sourceDocumentId: string): Promise<ViewerPayload> {
  const response = await fetch(corpusEndpoint("viewer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: sourceDocumentId }),
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

export function answerTextOrFallback(response: TjiptoAskResponse) {
  const answer = typeof response.answer === "string" ? response.answer.trim() : "";
  return answer || fallbackAnswer();
}

export function mapAskResponseToCitations(response: TjiptoAskResponse): Citation[] {
  return (response.supports ?? []).flatMap<Citation>((support, index) => {
    if (!support.support_id || support.linkable !== true || support.highlightable !== true || support.viewer_target?.can_resolve !== true) return [];
    const panelSection = support.panel_section as NonNullable<Citation["panelSection"]>;
    const pages = support.page_numbers ?? support.viewer_target.page_numbers ?? [];
    return [{
      id: index + 1,
      documentId: String(support.support_id),
      sourceDocumentId: support.source_document,
      documentTitle: fallbackDocumentTitle("uud"),
      regulationType: "UUD" as const,
      authorityKind: support.panel_section === "Kutipan Relevan"
        ? "legal_citation"
        : support.panel_section === "Catatan Sumber" ? "instrument_provenance" : "metadata_source",
      authorityLabel: support.display_label ?? panelSection,
      citationFinal: support.legal_citation_available === true,
      article: support.display_label ?? panelSection,
      pageNumber: Number(pages[0] ?? 1),
      excerpt: support.display_text ?? "",
      supportKind: support.support_kind,
      relevantQuoteEligible: support.relevant_quote_eligible === true,
      displayText: support.display_text ?? "",
      copyText: support.copy_text ?? support.display_text ?? "",
      layoutLines: support.layout_lines,
      viewerTarget: support.viewer_target as Record<string, unknown>,
      viewerRefId: String(support.support_id),
      sourceUrl: "",
      sourceDomain: support.source_role ?? "runtime",
      sourceRole: support.source_role,
      panelSection,
    }];
  });
}

export function mapAskResponseToDocumentSource(response: TjiptoAskResponse): Citation | null {
  const source = response.document_source;
  const target = source?.viewer_target?.target;
  if (response.answer_type !== "source_document" || !target) return null;
  return {
    id: 1,
    documentId: target,
    documentTitle: source.document_title ?? fallbackDocumentTitle("uud"),
    regulationType: "UUD",
    viewerMode: "document",
    pageNumber: 1,
    excerpt: "",
    sourceUrl: "",
    sourceRole: source.source_role,
    temporalContext: source.temporal_context,
    sourceStatusLabel: sourceStatusLabel(source.source_role, source.temporal_context),
  };
}

export function mapAskResponseToSupportItems(response: TjiptoAskResponse): {
  metadata: SupportItem[];
  structure: SupportItem[];
  trace: SupportItem[];
} {
  const grouped = (panel: string, kind: SupportItem["kind"]) => (response.supports ?? [])
    .filter((row) => row.panel_section === panel)
    .map((row, index) => ({
      id: String(row.support_id ?? `${kind}_${index}`),
      kind,
      label: row.display_label ?? panel,
      detail: row.display_text,
      clickable: row.linkable === true && row.highlightable === true && row.viewer_target?.can_resolve === true,
    }));
  return {
    metadata: grouped("Sumber Dokumen", "metadata"),
    structure: grouped("Struktur Dokumen", "structure"),
    trace: grouped("Catatan Sumber", "trace"),
  };
}

export function mapSearchResultToCitation(item: SearchResult, index: number): Citation | null {
  const target = item?.viewer_target?.target;
  if (!target || !item.snippet) return null;
  const pages = Array.isArray(item.page_numbers)
    ? item.page_numbers
    : Array.isArray(item.viewer_target?.page_numbers)
      ? item.viewer_target.page_numbers
      : [];
  const pageNumber = Number(pages[0] ?? 1);
  return {
    id: index + 1,
    documentId: target,
    viewerMode: item.status === "document" ? "document" : "evidence",
    viewerRefId: target,
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

function fallbackAuthorityKind(response: TjiptoAskResponse): Citation["authorityKind"] {
  if (
    response.route === "source_anomaly_explanation" &&
    response.answer_scope === "source_conflict_exact_provenance" &&
    Array.isArray(response.warnings) &&
    response.warnings.includes("source_conflict_not_final_legal_authority")
  ) {
    return "source_conflict_provenance";
  }
  return "legal_citation";
}

function fallbackAuthorityLabel(authorityKind: Citation["authorityKind"]) {
  return {
    legal_citation: "Sitasi hukum",
    metadata_source: "Metadata sumber",
    metadata_trace: "Metadata trace",
    source_conflict_provenance: "Jejak audit sumber",
    source_anomaly: "Source anomaly",
    structural_context: "Provenance struktural",
    instrument_provenance: "Instrument provenance",
  }[authorityKind ?? "legal_citation"];
}
