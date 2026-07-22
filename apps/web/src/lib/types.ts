export type CorpusId = string;
export type RegulationType = CorpusId;

export interface Citation {
  id: number;
  documentId: string;
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
  legalUnitId?: string;
  sourceDocumentId?: string;
  viewerRefId?: string;
  relationId?: string;
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
  sourceUrl: string;
  sourceDomain?: string;
  sourceRole?: string;
  temporalContext?: string;
  sourceStatusLabel?: string;
  relationSourceProofTextSpanIds?: string[];
  relationSourceProofBBoxRefs?: string[];
  relationTargetTextSpanIds?: string[];
  relationTargetBBoxRefs?: string[];
  relationTargetPrecision?: string;
  relationProof?: boolean;
  panelSection?: "Kutipan Relevan" | "Bukti Metadata" | "Struktur Dokumen" | "Catatan Sumber";
}

export interface LayoutLine {
  text: string;
  line_order: number;
  paragraph_id: string;
  alignment: "left" | "center" | "right" | "justify" | "unknown";
  indent: number;
  source_bbox_refs: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  metadataSupport?: SupportItem[];
  traceSupport?: SupportItem[];
  documentRelations?: SupportItem[];
  articleRelations?: SupportItem[];
  clarificationOptions?: { sourceRole?: string; label: string }[];
  status?: "streaming" | "complete";
  runtimeStatus?: string;
}

export interface SupportItem {
  id: string;
  label: string;
  detail?: string;
  kind: "metadata" | "trace" | "document_relation" | "article_relation";
  clickable?: boolean;
}

export interface ChatThread {
  id: string;
  title: string;
  group: "Today" | "Yesterday" | "Previous 7 Days" | "Older";
  active?: boolean;
}
