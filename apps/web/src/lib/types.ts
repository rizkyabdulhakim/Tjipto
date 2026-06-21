export type RegulationType = "UUD";

export interface BoundingBox {
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Citation {
  id: number;
  documentId: string;
  documentTitle: string;
  regulationType: RegulationType;
  article?: string;
  paragraph?: string;
  pageNumber: number;
  excerpt: string;
  sourceUrl: string;
  sourceDomain?: string;
  fileHash?: string;
  boundingBox?: BoundingBox;
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
