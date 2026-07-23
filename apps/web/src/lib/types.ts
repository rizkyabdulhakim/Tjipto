export type CorpusId = string;
export type RegulationType = CorpusId;

export interface Citation {
  id: number;
  publicTargetId: string;
  documentTitle: string;
  regulationType: RegulationType;
  authorityKind?:
    | "legal_citation"
    | "metadata_source"
    | "metadata_trace"
    | "source_conflict_provenance"
    | "source_anomaly"
    | "structural_context"
    | "instrument_provenance";
  authorityLabel?: string;
  citationFinal?: boolean;
  viewerMode?: "evidence" | "document";
  article?: string;
  paragraph?: string;
  pageNumber: number;
  excerpt: string;
  supportKind?: string;
  relevantQuoteEligible?: boolean;
  displayText?: string;
  copyText?: string;
  layoutLines?: LayoutLine[];
  viewerTarget?: Record<string, unknown>;
  sourceDomain?: string;
  sourceRole?: string;
  temporalContext?: string;
  sourceStatusLabel?: string;
  panelSection?: "Kutipan Relevan" | "Sumber Dokumen" | "Struktur Dokumen" | "Catatan Sumber";
}

export interface LayoutLine {
  text: string;
  line_order: number;
  paragraph_id: string;
  alignment: "left" | "center" | "right" | "justify" | "unknown";
  indent: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  metadataSupport?: SupportItem[];
  structuralSupport?: SupportItem[];
  traceSupport?: SupportItem[];
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
  members: SupportItem[];
}

export interface ChatThread {
  id: string;
  title: string;
  group: "Today" | "Yesterday" | "Previous 7 Days" | "Older";
  active?: boolean;
}
