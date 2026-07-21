import type { Citation, SupportItem } from "./types";
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
  citations?: CitationPayload[];
  historical_citations?: CitationPayload[];
  viewer_refs?: ViewerRefPayload[];
  metadata_support?: MetadataSupportPayload[];
  document_relations?: DocumentRelationPayload[];
  article_amendment_relations?: ArticleAmendmentRelationPayload[];
  trace_support?: TraceSupportPayload[];
  clarification_options?: ClarificationOptionPayload[];
}

export interface DocumentSourcePayload {
  source_document_id?: string;
  source_role?: string;
  temporal_context?: string;
  document_title?: string;
  viewer_target?: {
    action?: "open_document";
    source_document_id?: string;
  };
}

export interface ClarificationOptionPayload {
  source_role?: string;
  label?: string;
}

export interface MetadataSupportPayload {
  support_class?: "metadata_support" | "metadata_trace" | "exact_metadata_citation";
  authority_kind?: Citation["authorityKind"];
  authority_label?: string;
  citation_final?: boolean;
  field?: string;
  answer?: string;
  evidence_id?: string;
  source_document_id?: string;
  source_role?: string;
  page_numbers?: number[];
  citation_available?: boolean;
  viewer_highlightable?: boolean;
  viewer_ref?: ViewerRefPayload;
}

export interface DocumentRelationPayload {
  relation_id?: string;
  relation_type?: string;
  source_role?: string;
  source_document_id?: string;
  target_source_role?: string;
  target_document_id?: string;
  support_type?: string;
  reason?: string;
  highlightable?: boolean;
}

export interface ArticleAmendmentRelationPayload {
  relation_id?: string;
  relation_type?: string;
  source_document_id?: string;
  source_legal_unit_id?: string;
  source_legal_unit_role?: string;
  source_label?: string;
  source_reference?: string;
  source_reference_range?: [number, number];
  source_reference_range_kind?: "literal" | "contextual";
  source_role?: string;
  target_legal_unit_id?: string;
  target_label?: string;
  target_reference?: string;
  target_reference_range?: [number, number];
  target_reference_range_kind?: "literal" | "contextual";
  target_source_role?: string;
  evidence_id?: string;
  text_span_ids?: string[];
  bbox_refs?: string[];
  target_text_span_ids?: string[];
  target_bbox_refs?: string[];
  source_proof_text_span_ids?: string[];
  source_proof_bbox_refs?: string[];
  target_precision?: string;
  source_support_exact?: boolean;
  support_class?: string;
  trace_only_reason?: string;
  citation_available?: boolean;
  viewer_highlightable?: boolean;
}

export interface TraceSupportPayload {
  authority_kind?: Citation["authorityKind"];
  authority_label?: string;
  citation_final?: boolean;
  source_conflict_id?: string;
  relation_id?: string;
  relation_type?: string;
  source_role?: string;
  classification?: string;
  target_citation?: string;
  evidence_id?: string;
  support_class?: string;
  grounding_level?: string;
  citation_available?: boolean;
  viewer_highlightable?: boolean;
  failure_reason?: string;
}

export interface SearchResult {
  corpus_id: string;
  legal_unit_id?: string;
  evidence_id?: string;
  document_id?: string;
  citation_id?: string;
  viewer_ref_id?: string;
  source_document_id?: string;
  source_url?: string;
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
  evidence_id?: string;
  source_document_id?: string;
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

export async function getLegalViewerPayload(evidenceId: string, relationId?: string): Promise<ViewerPayload> {
  const response = await fetch(corpusEndpoint("viewer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evidence_id: evidenceId, ...(relationId ? { relation_id: relationId } : {}) }),
  });
  if (!response.ok) throw new Error(`${DEFAULT_CORPUS_ID} viewer returned ${response.status}`);
  return response.json();
}

export async function getDocumentViewerPayload(sourceDocumentId: string): Promise<ViewerPayload> {
  const response = await fetch(corpusEndpoint("viewer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_document_id: sourceDocumentId }),
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
  const citations: CitationPayload[] = [
    ...(Array.isArray(response.citations) ? response.citations : []),
    ...(Array.isArray(response.historical_citations) ? response.historical_citations : []),
    ...(response.metadata_support ?? [])
      .filter((item): item is MetadataSupportPayload & { evidence_id: string } => item?.evidence_id != null && item.viewer_ref?.can_resolve === true)
      .map((item) => ({
      evidence_id: item.evidence_id,
      source_document_id: item.source_document_id,
      source_role: item.source_role,
      page_numbers: item.page_numbers,
      authority_kind: item.authority_kind,
      authority_label: item.authority_label,
      citation_final: item.citation_final,
      viewer_ref: item.viewer_ref,
      citation: item.field ?? "Metadata",
      label: item.field ?? "Metadata",
      quoted_text: item.answer ?? item.field ?? "Metadata",
    })),
  ];
  const viewerRefs = Array.isArray(response.viewer_refs) ? response.viewer_refs : [];
  return citations.flatMap((item, index) => {
    if (!item?.evidence_id || !item?.quoted_text) return [];
    const viewer = item.viewer_ref ?? viewerRefs[index];
    if (viewer?.can_resolve !== true) return [];
    const relation = (response.article_amendment_relations ?? []).find((candidate) => candidate.evidence_id === item.evidence_id);
    const authorityKind = item.authority_kind ?? fallbackAuthorityKind(response);
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
      sourceDocumentId: item.source_document_id ?? viewer?.source_document_id,
      viewerRefId: viewer?.evidence_id,
      relationId: relation?.relation_id,
      documentTitle: item.document_title ?? fallbackDocumentTitle(item.corpus_id),
      regulationType: "UUD",
      authorityKind,
      authorityLabel: item.authority_label ?? fallbackAuthorityLabel(authorityKind),
      citationFinal: item.citation_final ?? authorityKind === "legal_citation",
      article: String(item.label ?? item.citation ?? "UUD"),
      pageNumber: Number.isFinite(pageNumber) ? pageNumber : 1,
      excerpt: String(item.quoted_text),
      sourceUrl: item.source_url ?? "",
      sourceDomain: item.source_role ?? item.corpus_id ?? "runtime",
      sourceRole: item.source_role,
      temporalContext: item.temporal_context,
      sourceStatusLabel: sourceStatusLabel(item.source_role, item.temporal_context),
      relationSourceProofTextSpanIds: viewer?.source_proof_text_span_ids,
      relationSourceProofBBoxRefs: viewer?.source_proof_bbox_refs,
      relationTargetTextSpanIds: viewer?.target_text_span_ids,
      relationTargetBBoxRefs: viewer?.target_bbox_refs,
      relationTargetPrecision: relation?.target_precision,
      relationProof: Array.isArray(viewer?.source_proof_bbox_refs) && viewer.source_proof_bbox_refs.length > 0,
    };
  });
}

export function mapAskResponseToDocumentSource(response: TjiptoAskResponse): Citation | null {
  const source = response.document_source;
  const sourceDocumentId = source?.source_document_id;
  if (response.answer_type !== "source_document" || !sourceDocumentId) return null;
  return {
    id: 1,
    documentId: sourceDocumentId,
    documentTitle: source.document_title ?? fallbackDocumentTitle("uud"),
    regulationType: "UUD",
    viewerMode: "document",
    sourceDocumentId,
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
  trace: SupportItem[];
  documentRelations: SupportItem[];
  articleRelations: SupportItem[];
} {
  return {
    metadata: (response.metadata_support ?? []).map((row, index) => ({
      id: String(row.evidence_id ?? `metadata_${index}`),
      kind: "metadata",
      label: row.authority_label ?? (row.authority_kind === "metadata_trace" ? "Metadata trace" : "Metadata sumber"),
      detail: [row.field, row.answer ?? (row.support_class === "exact_metadata_citation" ? "Metadata exact tersedia" : "Metadata trace-only")]
        .filter(Boolean)
        .join(" · "),
      clickable: false,
    })),
    trace: (response.trace_support ?? []).map((row, index) => ({
      id: String(row.source_conflict_id ?? row.relation_id ?? row.evidence_id ?? `trace_${index}`),
      kind: "trace",
      label: row.authority_label ?? String(row.target_citation ?? row.classification ?? row.relation_type ?? "trace_support"),
      detail: [
        row.classification,
        row.support_class === "source_conflict_trace" ? "Trace-only, tidak dapat di-highlight." : "Trace-only support.",
      ]
        .filter(Boolean)
        .join(" · "),
      clickable: false,
    })),
    documentRelations: (response.document_relations ?? []).map((row, index) => ({
      id: String(row.relation_id ?? `document_relation_${index}`),
      kind: "document_relation",
      label: String(row.relation_type ?? "document_relation"),
      detail: "Relasi tingkat dokumen, bukan sitasi atau highlight exact.",
      clickable: false,
    })),
    articleRelations: (response.article_amendment_relations ?? []).map((row, index) => ({
      id: String(row.relation_id ?? `article_relation_${index}`),
      kind: "article_relation",
      label: `${row.relation_type ?? "RELATION"}: ${row.source_reference ?? row.source_label ?? "source"} → ${row.target_reference ?? row.target_label ?? "target"}`,
      detail: row.support_class === "exact_article_relation"
        ? `Source proof exact (${row.source_proof_text_span_ids?.length ?? row.text_span_ids?.length ?? 0} spans / ${row.source_proof_bbox_refs?.length ?? row.bbox_refs?.length ?? 0} BBoxes); target ${row.target_precision ?? "shared_span"}. Evidence ${row.evidence_id ?? "unavailable"}.`
        : `Source-backed trace-only (${row.trace_only_reason ?? "precision not isolated"}; ${row.source_proof_text_span_ids?.length ?? row.text_span_ids?.length ?? 0} proof spans). Evidence ${row.evidence_id ?? "unavailable"}.`,
      clickable: row.viewer_highlightable === true,
    })),
  };
}

export function mapSearchResultToCitation(item: SearchResult, index: number): Citation | null {
  if (!item?.source_document_id || !item.snippet) return null;
  const pages = Array.isArray(item.page_numbers)
    ? item.page_numbers
    : Array.isArray(item.viewer_ref?.page_numbers)
      ? item.viewer_ref.page_numbers
      : [];
  const pageNumber = Number(pages[0] ?? 1);
  return {
    id: index + 1,
    documentId: String(item.evidence_id ?? item.document_id ?? item.source_document_id),
    viewerMode: item.status === "document" ? "document" : "evidence",
    legalUnitId: item.legal_unit_id,
    sourceDocumentId: item.source_document_id,
    viewerRefId: item.viewer_ref_id ?? item.viewer_ref?.evidence_id,
    documentTitle: item.document_title ?? fallbackDocumentTitle(item.corpus_id),
    regulationType: "UUD",
    article: String(item.label ?? item.citation ?? item.title ?? "UUD"),
    pageNumber: Number.isFinite(pageNumber) ? pageNumber : 1,
    excerpt: String(item.snippet),
    sourceUrl: item.source_url ?? "",
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
