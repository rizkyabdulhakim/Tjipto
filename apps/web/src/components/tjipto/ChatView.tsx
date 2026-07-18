import { useState, useEffect } from "react";
import { motion } from "motion/react";
import {
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCw,
  Share2,
  Volume2,
  FileText,
} from "lucide-react";
import type { ChatMessage, Citation, SupportItem } from "../../lib/types";
import { Composer } from "./Composer";

interface ChatViewProps {
  messages: ChatMessage[];
  onSubmit: (value: string) => void;
  isStreaming: boolean;
  onStop: () => void;
  onCitationClick: (citation: Citation) => void;
  activeCitationId?: number;
}

// Render markdown-lite: **bold** + paragraphs + inline [n] citations
function renderContent(
  content: string,
  citations: Citation[] | undefined,
  onCite: (c: Citation) => void,
  activeId: number | undefined,
) {
  const paragraphs = content.split(/\n\n+/);
  return paragraphs.map((para, pi) => {
    const parts: (string | { citation: Citation })[] = [];
    let last = 0;
    const regex = /\[(\d+)\]/g;
    let m: RegExpExecArray | null;
    while ((m = regex.exec(para)) !== null) {
      const match = m;
      if (match.index > last) parts.push(para.slice(last, match.index));
      const c = citations?.find((x) => x.id === Number(match[1]));
      if (c) parts.push({ citation: c });
      else parts.push(match[0]);
      last = match.index + match[0].length;
    }
    if (last < para.length) parts.push(para.slice(last));

    return (
      <p
        key={pi}
        className="my-3 first:mt-0 last:mb-0"
        style={{
          fontSize: 16,
          lineHeight: "26px",
          color: "var(--tj-text-primary)",
          fontWeight: 400,
        }}
      >
        {parts.map((p, i) => {
          if (typeof p === "string") {
            // bold
            const bold = p.split(/(\*\*[^*]+\*\*)/g).map((seg, j) => {
              if (seg.startsWith("**") && seg.endsWith("**")) {
                return (
                  <strong key={j} style={{ fontWeight: 600 }}>
                    {seg.slice(2, -2)}
                  </strong>
                );
              }
              return <span key={j}>{seg}</span>;
            });
            return <span key={i}>{bold}</span>;
          }
          const c = p.citation;
          const isActive = activeId === c.id;
          return (
            <CitationChip
              key={i}
              citation={c}
              active={isActive}
              onClick={() => onCite(c)}
            />
          );
        })}
      </p>
    );
  });
}

function CitationChip({
  citation,
  active,
  onClick,
}: {
  citation: Citation;
  active: boolean;
  onClick: () => void;
}) {
  const [hover, setHover] = useState(false);
  const location = legalUnitLabel(citation.article, citation.paragraph);
  return (
    <span className="relative inline-block align-baseline">
      <button
        data-citation-kind={citation.authorityKind ?? "legal_citation"}
        onClick={onClick}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        aria-label={`Sitasi ${citation.id}: ${citation.documentTitle}`}
        className={`inline-flex items-center justify-center align-baseline mx-[1px] rounded-[6px] border transition-all duration-150 ${
          active
            ? "bg-[var(--tj-accent)] text-white border-[var(--tj-accent)]"
            : "bg-[var(--tj-accent-soft)] text-[var(--tj-accent)] border-[var(--tj-accent-soft)] hover:border-[var(--tj-accent)]"
        }`}
        style={{
          fontSize: 11,
          fontWeight: 600,
          minWidth: 18,
          height: 18,
          padding: "0 5px",
          lineHeight: 1,
          transform: "translateY(-1px)",
        }}
      >
        {citation.id}
      </button>
      {hover && (
        <span
          className="absolute z-30 left-1/2 -translate-x-1/2 bottom-[calc(100%+8px)] w-[260px] rounded-lg border border-[var(--tj-border)] bg-[var(--tj-surface)] p-3 pointer-events-none"
          style={{ boxShadow: "var(--tj-shadow-panel)" }}
        >
          <span className="flex items-center gap-1.5 mb-1">
            <span
              className="inline-flex items-center px-1.5 h-[18px] rounded text-[10px] tracking-wide"
              style={{
                fontWeight: 600,
                background: "var(--tj-accent-soft)",
                color: "var(--tj-accent)",
              }}
            >
              {citation.authorityLabel ?? citation.regulationType.replace("_", " ")}
            </span>
            <span style={{ fontSize: 11, color: "var(--tj-text-muted)" }}>
              Halaman {citation.pageNumber}
            </span>
          </span>
          <span
            className="block mb-1"
            style={{ fontSize: 13, fontWeight: 600, color: "var(--tj-text-primary)", lineHeight: "18px" }}
          >
            {citation.documentTitle}
          </span>
          <span className="block" style={{ fontSize: 12, color: "var(--tj-text-secondary)" }}>
            {location}
          </span>
          {citation.citationFinal === false && (
            <span
              className="block mt-2"
              style={{ fontSize: 11, color: "var(--tj-text-muted)" }}
            >
              Bukan kesimpulan hukum final.
            </span>
          )}
          <span
            className="block mt-2 pt-2 border-t border-[var(--tj-border-subtle)]"
            style={{ fontSize: 11, color: "var(--tj-accent)", fontWeight: 500 }}
          >
            Klik untuk melihat panel bukti
          </span>
        </span>
      )}
    </span>
  );
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end mb-6">
      <div
        className="max-w-[85%] sm:max-w-[80%] rounded-[20px] rounded-tr-md px-4 py-2.5 bg-[var(--tj-user-bubble)]"
        style={{
          fontSize: 16,
          lineHeight: "24px",
          color: "var(--tj-text-primary)",
        }}
      >
        {content}
      </div>
    </div>
  );
}

function AssistantMessage({
  message,
  onSubmit,
  onCitationClick,
  activeCitationId,
}: {
  message: ChatMessage;
  onSubmit: (value: string) => void;
  onCitationClick: (c: Citation) => void;
  activeCitationId?: number;
}) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    try {
      const textArea = document.createElement("textarea");
      textArea.value = message.content;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch (err) {
      console.error("Fallback copy failed", err);
    }
  };
  return (
    <div className="mb-10 last:mb-20">
      <div className="flex flex-col gap-2">
        <div
          className="flex items-center gap-2 mb-1 opacity-40 select-none"
          style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", color: "var(--tj-text-primary)" }}
        >
          ANALISIS TJIPTO
          {message.runtimeStatus && (
            <span data-runtime-status={message.runtimeStatus} className="rounded-md border border-[var(--tj-border-subtle)] px-1.5 py-0.5">
              {message.runtimeStatus}
            </span>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="tj-assistant-content">
            {renderContent(
              message.content,
              message.citations,
              onCitationClick,
              activeCitationId,
            )}
            {message.status === "streaming" && (
              <span
                className="inline-block w-[2px] h-[18px] align-middle ml-1 tj-cursor"
                style={{ background: "var(--tj-accent)" }}
              />
            )}
          </div>

            {message.status !== "streaming" && message.citations && (
            <CitationFooter
              citations={message.citations}
              onClick={onCitationClick}
              activeId={activeCitationId}
            />
            )}

            {message.status !== "streaming" && message.clarificationOptions?.length ? (
              <div
                data-clarification-options="true"
                className="mt-4 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)] px-3.5 py-2.5"
              >
                <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.04em", color: "var(--tj-text-secondary)" }}>
                  PILIH KONTEKS SUMBER
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {message.clarificationOptions.map((option) => (
                    <button
                      key={option.sourceRole ?? option.label}
                      type="button"
                      data-clarification-option={option.sourceRole ?? option.label}
                      onClick={() => onSubmit(option.label)}
                      className="rounded-lg border border-[var(--tj-border-subtle)] px-2.5 py-1.5 text-xs hover:bg-[var(--tj-surface-hover)]"
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

          {message.status !== "streaming" && (
            <SupportFooter
              metadataSupport={message.metadataSupport}
              traceSupport={message.traceSupport}
              documentRelations={message.documentRelations}
              articleRelations={message.articleRelations}
            />
          )}

          {message.status !== "streaming" && (
            <div className="mt-4 flex items-center gap-1 -ml-1.5">
              <IconButton icon={Copy} label={copied ? "Copied" : "Copy"} onClick={copy} />
              <IconButton icon={ThumbsUp} label="Bermanfaat" />
              <IconButton icon={ThumbsDown} label="Kurang tepat" />
              <IconButton icon={Volume2} label="Baca bersuara" />
              <IconButton icon={RotateCw} label="Ulangi" />
              <IconButton icon={Share2} label="Bagikan" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function IconButton({
  icon: Icon,
  label,
  onClick,
}: {
  icon: typeof Copy;
  label: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      aria-label={label}
      className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)] hover:text-[var(--tj-text-primary)] transition-colors"
    >
      <Icon size={15} />
    </button>
  );
}

function CitationFooter({
  citations,
  onClick,
  activeId,
}: {
  citations: Citation[];
  onClick: (c: Citation) => void;
  activeId?: number;
}) {
  const hasProvenance = citations.some((c) => c.citationFinal === false);
  return (
    <div data-citation-footer="true" className="mt-5 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)] overflow-hidden">
      <div className="flex items-center gap-2 px-3.5 py-2.5 border-b border-[var(--tj-border-subtle)]">
        <FileText size={13} className="text-[var(--tj-text-secondary)]" />
        <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.02em", color: "var(--tj-text-secondary)" }}>
          {hasProvenance ? "PROVENANSI SUMBER" : "SUMBER"} · {citations.length} sitasi
        </span>
      </div>
      {hasProvenance && (
        <div
          data-provenance-note="true"
          className="px-3.5 py-2 border-b border-[var(--tj-border-subtle)]"
          style={{ fontSize: 12, color: "var(--tj-text-muted)" }}
        >
          Bukan kesimpulan hukum final. Entri di bawah ini ditampilkan sebagai provenance sumber yang dapat diaudit.
        </div>
      )}
      <ul>
        {citations.map((c) => (
          <li key={c.id}>
            <button
              data-citation-kind={c.authorityKind ?? "legal_citation"}
              onClick={() => onClick(c)}
              className={`w-full flex items-start gap-3 px-3.5 py-2.5 text-left transition-colors border-b border-[var(--tj-border-subtle)] last:border-b-0 ${
                activeId === c.id
                  ? "bg-[var(--tj-accent-soft)]"
                  : "hover:bg-[var(--tj-surface-hover)]"
              }`}
            >
              <span
                className="shrink-0 inline-flex items-center justify-center mt-0.5"
                style={{
                  minWidth: 18,
                  height: 22,
                  fontSize: 12,
                  fontWeight: 600,
                  color: "var(--tj-accent)",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {c.id}
              </span>
              <span className="flex-1 min-w-0">
                <span
                  className="block truncate"
                  style={{ fontSize: 13, fontWeight: 500, color: "var(--tj-text-primary)" }}
                >
                  {c.documentTitle}
                </span>
                {c.citationFinal === false && (
                  <span
                    className="block truncate mt-0.5"
                    style={{ fontSize: 11, color: "var(--tj-accent)", fontWeight: 600 }}
                  >
                    {c.authorityLabel ?? "Provenansi sumber"}
                  </span>
                )}
                <span
                  className="block truncate mt-0.5"
                  style={{ fontSize: 12, color: "var(--tj-text-muted)" }}
                >
                  {legalUnitLabel(c.article, c.paragraph)}
                  {" "}· hal. {c.pageNumber} · {c.sourceDomain}
                </span>
              </span>
              <FileText size={13} className="text-[var(--tj-text-muted)] shrink-0 mt-1" />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}


function legalUnitLabel(article?: string, paragraph?: string) {
  const base = article || "UUD";
  const knownLabel = /^(pasal|bab|aturan|pembukaan)\b/i.test(base) || base.includes(" / ");
  const label = knownLabel ? base : `Pasal ${base}`;
  return paragraph ? `${label} ayat (${paragraph})` : label;
}

function SupportFooter({
  metadataSupport,
  traceSupport,
  documentRelations,
  articleRelations,
}: {
  metadataSupport?: SupportItem[];
  traceSupport?: SupportItem[];
  documentRelations?: SupportItem[];
  articleRelations?: SupportItem[];
}) {
  const groups = [
    ["metadata-support", "DUKUNGAN METADATA", metadataSupport],
    ["trace-support", "TRACE-ONLY", traceSupport],
    ["document-relations", "RELASI DOKUMEN", documentRelations],
    ["article-relations", "BUKTI RELASI PASAL", articleRelations],
  ] as const;
  const visible = groups.filter(([, , rows]) => rows?.length);
  if (!visible.length) return null;
  return (
    <div className="mt-4 space-y-2">
      {visible.map(([testId, title, rows]) => (
        <div
          key={testId}
          data-support-kind={testId}
          className="rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)] px-3.5 py-2.5"
        >
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.04em", color: "var(--tj-text-secondary)" }}>
            {title}
          </div>
          <ul className="mt-1 space-y-1">
            {rows?.map((row) => (
              <li
                key={row.id}
                data-support-clickable={row.clickable ? "true" : "false"}
                style={{ fontSize: 12, color: "var(--tj-text-muted)" }}
              >
                <span style={{ color: "var(--tj-text-primary)", fontWeight: 600 }}>{row.label}</span>
                {row.detail ? ` · ${row.detail}` : ""}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export function ChatView({
  messages,
  onSubmit,
  isStreaming,
  onStop,
  onCitationClick,
  activeCitationId,
}: ChatViewProps) {
  // auto-scroll
  useEffect(() => {
    const el = document.getElementById("tj-scroll-area");
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      <div
        id="tj-scroll-area"
        className="flex-1 overflow-y-auto tj-scroll"
      >
        <div className="max-w-[760px] mx-auto px-4 sm:px-6 pt-6 sm:pt-10 pb-6 sm:pb-8">
          {messages.map((m, i) =>
            m.role === "user" ? (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: i * 0.02 }}
              >
                <UserMessage content={m.content} />
              </motion.div>
            ) : (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}
              >
                <AssistantMessage
                  message={m}
                  onSubmit={onSubmit}
                  onCitationClick={onCitationClick}
                  activeCitationId={activeCitationId}
                />
              </motion.div>
            ),
          )}
        </div>
      </div>
      <div className="shrink-0 pt-2 pb-6 relative z-10">
        <div className="absolute inset-0 bg-gradient-to-t from-[var(--tj-bg)] via-[var(--tj-bg)] to-transparent opacity-60 pointer-events-none" />
        <Composer onSubmit={onSubmit} isStreaming={isStreaming} onStop={onStop} />
      </div>
    </div>
  );
}
