export type RegulationType = "UUD";

export interface Citation {
  id: number;
  documentId: string;
  documentTitle: string;
  regulationType: RegulationType;
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
  status?: "streaming" | "complete";
  runtimeStatus?: string;
}

export interface ChatThread {
  id: string;
  title: string;
  group: "Today" | "Yesterday" | "Previous 7 Days" | "Older";
  active?: boolean;
}
