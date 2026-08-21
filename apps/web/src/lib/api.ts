import type { AuthorityKind, Citation, SupportGroup } from "./types";
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

export interface SupportPayload {
  public_support_id: string;
  authority_kind: AuthorityKind;
  citation_final: boolean;
  support_kind: string;
  fact_kind: string;
  label: string;
  role_label?: string | null;
  text: string;
  source_label?: string;
  source_status_label?: string;
  page_numbers: number[];
  viewer_target: ViewerTargetPayload;
  citation?: { number: number; text: string; official_url: string; citation_final: boolean } | null;
}

export interface SupportGroupPayload {
  public_group_id: string;
  label: string;
  summary: string;
  member_count: number;
  group_kind: "document_metadata" | "entity_occurrences" | "role_members";
  members: SupportPayload[];
}

export interface TjiptoAskResponse {
  kind: "answer" | "document" | "documents" | "unavailable";
  status: string;
  answer?: string;
  operation?: string;
  source_scopes?: { label: string }[];
  sufficiency?: { status: "complete" | "partial" | "insufficient"; missing_requirement_ids?: string[] };
  original_query?: string;
  document?: LegalDocumentPayload & { label?: string; source_status_label?: string };
  documents?: LegalDocumentPayload[];
  supports?: SupportPayload[];
  support_groups?: SupportGroupPayload[];
  clarification?: { id: string; missing_dimensions: string[] };
}

export interface LegalDocumentPayload {
  legal_identity?: string;
  title?: string;
  legal_status?: string;
  document_role?: string;
  issuer?: string;
  establishment_date?: string | null;
  promulgation_date?: string | null;
  effective_date?: string | null;
  publication?: string | null;
  official_url?: string;
  relations?: LegalRelationPayload[];
  provision_effects?: ProvisionEffectPayload[];
  source_annotations?: SourceAnnotationPayload[];
  official_title_conflict?: OfficialValueConflictPayload;
  viewer_target?: ViewerTargetPayload;
}

export type SearchResult = LegalDocumentPayload;

export interface CatalogFacet {
  name: string;
  label: string;
  options: { value: string; label: string; count: number }[];
}

export interface CatalogResponse {
  kind: "catalog";
  status: string;
  total: number;
  applied_filters: Record<string, string>;
  facets: CatalogFacet[];
  results: SearchResult[];
}

export interface ViewerPayload extends LegalDocumentPayload {
  status: string;
  citation?: string;
  quoted_text?: string;
  source_status_label?: string;
  page_numbers?: number[];
  bbox_rectangles?: (PdfBBox & { public_rectangle_id?: string; bbox_precision?: "exact" | "coarse" | "page_grounded_only"; viewer_highlightable?: boolean })[];
  viewer_highlightable?: boolean;
  pdf_access_available?: boolean;
  rendering_available?: boolean;
  pdf?: { mime_type?: string; access_url?: string };
  document?: LegalDocumentPayload;
  document_type?: string;
  number?: string;
  year?: string;
}

export interface LegalRelationPayload {
  label: string;
  relation_type: string;
  source: string;
  target: string;
  direction: string;
  verification_state: string;
  source_reference?: string;
}

export interface ProvisionEffectPayload {
  label: string;
  target: string;
  verification_state: string;
  source_reference?: string;
  page_number?: number;
}

export interface SourceAnnotationPayload {
  label: string;
  text: string;
  source_reference?: string;
  page_number?: number;
  viewer_target?: ViewerTargetPayload;
}

export interface OfficialValueConflictPayload {
  state: string;
  kind: string;
  values: {
    value: string;
    source_authority?: string;
    source_reference: string;
    verified_at: string;
  }[];
  reviewer_decision?: string;
  legal_basis?: string;
}

export interface BookmarkPointer {
  public_bookmark_id: string;
  public_target_id?: string;
  note?: string;
  created_at: string;
  status: string;
  document?: LegalDocumentPayload;
}

export async function askLegal(query: string, clarification?: { id: string }): Promise<TjiptoAskResponse> {
  return request("ask", clarification
    ? { query, clarification_id: clarification.id, clarification_answer: query }
    : { query });
}

export async function searchLegal(query: string, filters: Record<string, string> = {}): Promise<CatalogResponse> {
  return catalogRequest<CatalogResponse>("search", { query, limit: 10, filters });
}

export async function getCatalogFacets(): Promise<CatalogFacet[]> {
  const body = await catalogRequest<{ facets?: CatalogFacet[] }>("facets", {});
  return Array.isArray(body.facets) ? body.facets : [];
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

export async function deleteLegalBookmark(publicBookmarkId: string): Promise<boolean> {
  const response = await fetch(corpusEndpoint("bookmarks"), {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bookmark: publicBookmarkId }),
  });
  if (!response.ok) return false;
  const body = await response.json();
  return body.status === "deleted";
}

export async function getLegalViewerPayload(publicTargetId: string, catalog = false): Promise<ViewerPayload> {
  const payload = await (catalog
    ? catalogRequest<ViewerPayload>("viewer", { target: publicTargetId })
    : request<ViewerPayload>("viewer", { target: publicTargetId }));
  return payload.document ? { ...payload, ...payload.document } : payload;
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

async function catalogRequest<T>(action: string, body: object): Promise<T> {
  if (!API_BASE) throw new Error("missing_runtime_configuration");
  const response = await fetch(`${API_BASE}/legal/catalog/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`service returned ${response.status}`);
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
  const source = response.document;
  return source ? mapDocumentToCitation(source, 1, source.label, false) : null;
}

export function mapAskResponseToDocumentSources(response: TjiptoAskResponse): Citation[] {
  return (response.documents ?? []).flatMap((source, index) => {
    const citation = mapDocumentToCitation(source, index + 1, undefined, true);
    return citation ? [citation] : [];
  });
}

function mapDocumentToCitation(
  source: LegalDocumentPayload,
  id: number,
  preferredTitle?: string,
  includeRole = false,
): Citation | null {
  const target = source.viewer_target?.public_target_id;
  if (!target) return null;
  return {
    id,
    publicTargetId: target,
    documentTitle: preferredTitle ?? source.title ?? source.legal_identity ?? "Dokumen sumber",
    regulationType: "legal",
    authorityKind: "document_source",
    authorityLabel: includeRole ? source.document_role ?? "Sumber dokumen" : "Sumber dokumen",
    citationText: "Buka PDF sumber",
    viewerMode: "document",
    pageNumber: Number(source.viewer_target?.page_numbers?.[0] ?? 1),
    excerpt: includeRole ? source.title ?? "" : "",
    sourceStatusLabel: includeRole ? source.document_role : undefined,
  };
}

export function mapAskResponseToSupportGroups(response: TjiptoAskResponse): SupportGroup[] {
  const grouped = (response.support_groups ?? []).flatMap((group) => {
    const first = group.members[0];
    if (!first) return [];
    const kind = supportGroupKind(first.authority_kind);
    if (!kind) return [];
    return [{
      id: group.public_group_id,
      title: group.label,
      summary: group.summary,
      kind,
      groupKind: group.group_kind,
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
  const groupedIds = new Set((response.support_groups ?? []).flatMap((group) => group.members.map((member) => member.public_support_id)));
  const atomic = (response.supports ?? []).flatMap((support) => {
    const kind = supportGroupKind(support.authority_kind);
    if (groupedIds.has(support.public_support_id) || !kind) return [];
    return [{
      id: support.public_support_id,
      title: supportGroupTitle(kind),
      summary: support.source_label ?? support.label,
      kind,
      members: [{ id: support.public_support_id, publicTargetId: support.viewer_target.public_target_id ?? undefined, label: support.label, detail: support.text, kind, clickable: support.viewer_target.can_resolve === true }],
    }];
  });
  return [...grouped, ...atomic];
}

export function mapSearchResultToCitation(item: SearchResult, index: number): Citation | null {
  const target = item.viewer_target?.public_target_id;
  if (!target) return null;
  return {
    id: index + 1,
    publicTargetId: target,
    documentTitle: item.legal_identity ?? item.title ?? "Identitas belum diverifikasi",
    regulationType: "Dokumen hukum",
    viewerMode: "catalog",
    pageNumber: 1,
    excerpt: item.title ?? "",
    legalStatus: item.legal_status,
    documentRole: item.document_role,
    establishmentDate: item.establishment_date ?? undefined,
    officialUrl: item.official_url,
    issuer: item.issuer,
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
    authorityKind: support.authority_kind,
    authorityLabel: support.label,
    citationFinal: support.citation_final,
    citationNumber: support.citation?.number,
    citationText: support.citation?.text,
    article: support.label,
    pageNumber: Number(support.page_numbers[0] ?? 1),
    excerpt: support.text,
    supportKind: support.support_kind,
    factKind: support.fact_kind,
    viewerTarget: support.viewer_target as Record<string, unknown>,
    sourceStatusLabel: support.source_status_label,
  }];
}

function supportGroupKind(authority: AuthorityKind): SupportGroup["kind"] | null {
  if (authority === "legal_citation") return null;
  if (authority === "metadata_source" || authority === "metadata_trace") return "metadata";
  if (authority === "structural_context") return "structure";
  return "trace";
}

function supportGroupTitle(kind: SupportGroup["kind"]) {
  return kind === "metadata" ? "Sumber Dokumen" : kind === "structure" ? "Struktur Dokumen" : "Catatan Sumber";
}

export function legalReferenceLabel(article?: string, paragraph?: string) {
  const label = article || "Ketentuan";
  return paragraph ? `${label} ayat (${paragraph})` : label;
}
