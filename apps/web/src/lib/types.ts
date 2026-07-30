export type CorpusId = string;
export type RegulationType = CorpusId;
export type AuthorityKind =
  | "legal_citation"
  | "metadata_source"
  | "metadata_trace"
  | "source_conflict_provenance"
  | "source_anomaly"
  | "structural_context"
  | "instrument_provenance"
  | "source_text";

export interface Citation {
  id: number;
  publicTargetId: string;
  documentTitle: string;
  regulationType: RegulationType;
  authorityKind?: AuthorityKind;
  authorityLabel?: string;
  citationFinal?: boolean;
  viewerMode?: "evidence" | "document";
  article?: string;
  paragraph?: string;
  pageNumber: number;
  excerpt: string;
  supportKind?: string;
  factKind?: string;
  viewerTarget?: Record<string, unknown>;
  sourceDomain?: string;
  sourceRole?: string;
  temporalContext?: string;
  sourceStatusLabel?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  supportGroups?: SupportGroup[];
  clarificationOptions?: { sourceRole?: string; label: string }[];
  clarificationQuery?: string;
  status?: "streaming" | "complete";
  runtimeStatus?: string;
}

export interface SupportItem {
  id: string;
  publicTargetId?: string;
  label: string;
  detail?: string;
  kind: "metadata" | "structure" | "trace";
  clickable?: boolean;
}

export interface SupportGroup {
  id: string;
  title: string;
  summary: string;
  kind: SupportItem["kind"];
  groupKind?: "document_metadata" | "entity_occurrences" | "role_members" | "atomic";
  members: SupportItem[];
}

export interface ChatThread {
  id: string;
  title: string;
  group: "Today" | "Yesterday" | "Previous 7 Days" | "Older";
  active?: boolean;
}
