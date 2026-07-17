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
    | "instrument_provenance";
  authorityLabel?: string;
  citationFinal?: boolean;
  viewerMode?: "evidence" | "document";
  legalUnitId?: string;
  sourceDocumentId?: string;
  viewerRefId?: string;
  article?: string;
  paragraph?: string;
  pageNumber: number;
  excerpt: string;
  sourceUrl: string;
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
  metadataSupport?: SupportItem[];
  traceSupport?: SupportItem[];
  documentRelations?: SupportItem[];
  articleRelations?: SupportItem[];
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
