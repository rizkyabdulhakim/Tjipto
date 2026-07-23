import type { Citation, LayoutLine, SupportGroup, SupportItem } from "./types";
import type { PdfBBox } from "./pdfBBox";

declare global {
  interface Window {
    __TJIPTO_RUNTIME_CONFIG__?: { apiBase?: string; corpusId?: string };
  }
}

const DOCUMENT_CONFIG = typeof document === "undefined" ? undefined : document.documentElement.dataset;
const RUNTIME_CONFIG = typeof window === "undefined" ? undefined : window.__TJIPTO_RUNTIME_CONFIG__;
const API_BASE = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TJIPTO_API_BASE
  ?? RUNTIME_CONFIG?.apiBase
  ?? DOCUMENT_CONFIG?.tjiptoApiBase;
const CORPUS_ID = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TJIPTO_CORPUS_ID
  ?? RUNTIME_CONFIG?.corpusId
  ?? DOCUMENT_CONFIG?.tjiptoCorpus;

function corpusEndpoint(action: string) {
  if (!API_BASE || !CORPUS_ID) throw new Error("missing_runtime_configuration");
  return `${API_BASE}/legal/${encodeURIComponent(CORPUS_ID)}/${action}`;
}

export interface ViewerTargetPayload {
  action?: string;
  public_target_id?: string | null;
  page_numbers?: number[];
  can_resolve?: boolean;
}

export interface LayoutPayloadLine {
  text: string;
  line_order: number;
  paragraph_id: string;
  alignment: LayoutLine["alignment"];
  indent: number;
}

export interface SupportPayload {
  public_support_id: string;
  support_kind: string;
  panel_section: "Kutipan Relevan" | "Sumber Dokumen" | "Struktur Dokumen" | "Catatan Sumber";
  fact_kind: string;
  label: string;
  role_label?: string | null;
  text: string;
  layout_lines: LayoutPayloadLine[];
  copy_text: string;
  source_label?: string;
  source_role?: string;
  page_numbers: number[];
  legal_citation_available: boolean;
  relevant_quote_eligible: boolean;
  viewer_target: ViewerTargetPayload;
}

export interface SupportGroupPayload {
  public_group_id: string;
  panel_section: SupportPayload["panel_section"];
  label: string;
  summary: string;
  member_count: number;
  members: SupportPayload[];
}

export interface TjiptoAskResponse {
  status: string;
  answer?: string;
  answer_scope?: string;
  reason?: string | null;
  document_source?: { label?: string; source_role?: string; viewer_target?: ViewerTargetPayload };
  clarification_options?: { source_role?: string; label?: string }[];
  supports?: SupportPayload[];
  support_groups?: SupportGroupPayload[];
}

export interface SearchResult {
  title?: string;
  label?: string;
  snippet?: string;
  source_role?: string;
  page_numbers?: number[];
  viewer_target?: ViewerTargetPayload;
}

export interface ViewerPayload {
  status: string;
  citation?: string;
  quoted_text?: string;
  source_role?: string;
  source_status_label?: string;
  page_numbers?: number[];
  bbox_rectangles?: (PdfBBox & { public_rectangle_id?: string; bbox_precision?: "exact" | "coarse" | "page_grounded_only"; viewer_highlightable?: boolean })[];
  viewer_highlightable?: boolean;
  pdf_access_available?: boolean;
  rendering_available?: boolean;
  reason?: string | null;
  pdf?: { mime_type?: string; access_url?: string };
}

export interface BookmarkPointer {
  public_bookmark_id: string;
  public_target_id?: string;
  note?: string;
  created_at: string;
  status: string;
}

export async function askLegal(query: string, filters?: { source_role: string }): Promise<TjiptoAskResponse> {
  return request("ask", { query, ...(filters ? { filters } : {}) });
}

export async function searchLegal(query: string): Promise<SearchResult[]> {
  const body = await request<{ results?: SearchResult[] }>("search", { query, limit: 5 });
  return Array.isArray(body.results) ? body.results : [];
}

export async function listLegalBookmarks(): Promise<{ bookmarks: BookmarkPointer[] }> {
  const response = await fetch(corpusEndpoint("bookmarks"));
  if (!response.ok) throw new Error(`runtime returned ${response.status}`);
  const body = await response.json();
  return { bookmarks: Array.isArray(body.bookmarks) ? body.bookmarks : [] };
}

export async function saveLegalBookmark(publicTargetId: string): Promise<BookmarkPointer | null> {
  const body = await request<{ bookmark?: BookmarkPointer }>("bookmarks", { target: publicTargetId });
  return body.bookmark ?? null;
}

export async function getLegalViewerPayload(publicTargetId: string): Promise<ViewerPayload> {
  return request("viewer", { target: publicTargetId });
}

async function request<T = TjiptoAskResponse>(action: string, body: object): Promise<T> {
  const response = await fetch(corpusEndpoint(action), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`runtime returned ${response.status}`);
  return response.json();
}

export function pdfAccessUrl(viewer: ViewerPayload): string | null {
  const accessUrl = viewer.pdf?.access_url;
  return accessUrl && API_BASE ? new URL(accessUrl, API_BASE).toString() : null;
}

export function fallbackAnswer() {
  return "Bukti tidak cukup atau database belum tersedia dalam korpus terverifikasi saat ini.";
}

export function answerTextOrFallback(response: TjiptoAskResponse) {
  return response.answer?.trim() || fallbackAnswer();
}

export function mapAskResponseToCitations(response: TjiptoAskResponse): Citation[] {
  return (response.supports ?? []).flatMap((support, index) => mapSupportToCitation(support, index));
}

export function mapAskResponseToDocumentSource(response: TjiptoAskResponse): Citation | null {
  const source = response.document_source;
  const target = source?.viewer_target?.public_target_id;
  if (!target) return null;
  return {
    id: 1,
    publicTargetId: target,
    documentTitle: source.label ?? "Dokumen sumber",
    regulationType: "legal",
    viewerMode: "document",
    pageNumber: Number(source.viewer_target?.page_numbers?.[0] ?? 1),
    excerpt: "",
    sourceRole: source.source_role,
  };
}

export function mapAskResponseToSupportItems(response: TjiptoAskResponse): { metadata: SupportItem[]; structure: SupportItem[]; trace: SupportItem[] } {
  const supports = response.supports ?? [];
  const rows = (section: SupportPayload["panel_section"], kind: SupportItem["kind"]) => supports
    .filter((support) => support.panel_section === section)
    .map((support) => ({
      id: support.public_support_id,
      publicTargetId: support.viewer_target.public_target_id ?? undefined,
      label: support.label,
      detail: support.text,
      kind,
      clickable: support.viewer_target.can_resolve === true,
    }));
  return {
    metadata: rows("Sumber Dokumen", "metadata"),
    structure: rows("Struktur Dokumen", "structure"),
    trace: rows("Catatan Sumber", "trace"),
  };
}

export function mapAskResponseToSupportGroups(response: TjiptoAskResponse): SupportGroup[] {
  return (response.support_groups ?? []).flatMap((group) => {
    const kind = group.panel_section === "Sumber Dokumen" ? "metadata" : group.panel_section === "Struktur Dokumen" ? "structure" : group.panel_section === "Catatan Sumber" ? "trace" : null;
    if (!kind) return [];
    return [{
      id: group.public_group_id,
      title: group.label,
      summary: group.summary,
      kind,
      members: group.members.map((support) => ({
        id: support.public_support_id,
        publicTargetId: support.viewer_target.public_target_id ?? undefined,
        label: support.label,
        detail: support.text,
        kind,
        clickable: support.viewer_target.can_resolve === true,
      })),
    }];
  });
}

export function mapSearchResultToCitation(item: SearchResult, index: number): Citation | null {
  const target = item.viewer_target?.public_target_id;
  if (!target || !item.snippet) return null;
  return {
    id: index + 1,
    publicTargetId: target,
    documentTitle: item.title ?? "Dokumen sumber",
    regulationType: "legal",
    viewerMode: "evidence",
    article: item.label,
    pageNumber: Number(item.page_numbers?.[0] ?? item.viewer_target?.page_numbers?.[0] ?? 1),
    excerpt: item.snippet,
    sourceRole: item.source_role,
  };
}

function mapSupportToCitation(support: SupportPayload, index: number): Citation[] {
  const target = support.viewer_target.public_target_id;
  if (!target || support.viewer_target.can_resolve !== true) return [];
  return [{
    id: index + 1,
    publicTargetId: target,
    documentTitle: support.source_label ?? "Dokumen sumber",
    regulationType: "legal",
    authorityKind: support.panel_section === "Kutipan Relevan" ? "legal_citation" : undefined,
    authorityLabel: support.label,
    citationFinal: support.legal_citation_available,
    article: support.label,
    pageNumber: Number(support.page_numbers[0] ?? 1),
    excerpt: support.text,
    supportKind: support.support_kind,
    relevantQuoteEligible: support.relevant_quote_eligible,
    displayText: support.text,
    copyText: support.copy_text,
    layoutLines: support.layout_lines,
    viewerTarget: support.viewer_target as Record<string, unknown>,
    sourceRole: support.source_role,
    panelSection: support.panel_section,
  }];
}
