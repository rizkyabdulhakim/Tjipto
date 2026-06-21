import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  X,
  Minus,
  Plus,
  Maximize2,
  Copy,
  ChevronLeft,
  ChevronRight,
  Check,
  FileText,
  Hash,
  Scale,
} from "lucide-react";
import type { Citation } from "../../lib/types";

interface EvidencePanelProps {
  citation: Citation | null;
  allCitations: Citation[];
  onClose: () => void;
  onSelect: (c: Citation) => void;
}

export function EvidencePanel({
  citation,
  allCitations,
  onClose,
  onSelect,
}: EvidencePanelProps) {
  return (
    <AnimatePresence>
      {citation && (
        <>
          {/* Desktop side-by-side panel */}
          <motion.aside
            key="ev-desktop"
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.2, 0.8, 0.2, 1] }}
            className="hidden lg:flex flex-col w-[440px] shrink-0 h-full bg-[var(--tj-surface)]/60 backdrop-blur-3xl border-l border-[var(--tj-glass-border)] shadow-2xl z-10"
          >
            <EvidenceContent
              citation={citation}
              allCitations={allCitations}
              onClose={onClose}
              onSelect={onSelect}
            />
          </motion.aside>

          {/* Tablet / mobile overlay */}
          <motion.div
            key="ev-overlay-bg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={onClose}
            className="lg:hidden fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
          />
          <motion.aside
            key="ev-overlay"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1] }}
            className="lg:hidden fixed inset-y-0 right-0 z-50 flex flex-col w-full sm:w-[460px] sm:max-w-[95vw] bg-[var(--tj-surface)]/80 backdrop-blur-3xl border-l border-[var(--tj-glass-border)] shadow-2xl"
          >
            <EvidenceContent
              citation={citation}
              allCitations={allCitations}
              onClose={onClose}
              onSelect={onSelect}
            />
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function EvidenceContent({
  citation,
  allCitations,
  onClose,
  onSelect,
}: {
  citation: Citation;
  allCitations: Citation[];
  onClose: () => void;
  onSelect: (c: Citation) => void;
}) {
  const idx = allCitations.findIndex((c) => c.id === citation.id);
  const prev = allCitations[idx - 1];
  const next = allCitations[idx + 1];
  const [zoom, setZoom] = useState(100);
  const [copied, setCopied] = useState(false);
  const location = legalUnitLabel(citation.article, citation.paragraph);

  const copyExcerpt = () => {
    try {
      const textArea = document.createElement("textarea");
      textArea.value = citation.excerpt;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.error("Fallback copy failed", err);
    }
  };

  return (
    <>
      {/* HEADER */}
      <header className="px-6 pt-5 pb-4 border-b border-[var(--tj-border-subtle)] shrink-0">
        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1.5 min-w-0">
            <div className="flex items-center gap-2">
              <span
                className="inline-flex items-center px-2 h-[20px] rounded-md tracking-wider shrink-0"
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  background: "var(--tj-accent-soft)",
                  color: "var(--tj-accent)",
                  textTransform: "uppercase",
                }}
              >
                {citation.regulationType.replace("_", " ")}
              </span>
              <span
                className="inline-flex items-center justify-center rounded-lg shrink-0"
                style={{
                  width: 20,
                  height: 20,
                  background: "var(--tj-accent)",
                  color: "#fff",
                  fontSize: 11,
                  fontWeight: 700,
                }}
              >
                {citation.id}
              </span>
            </div>
            <h2
              className="tracking-tight"
              style={{
                fontSize: 17,
                lineHeight: "24px",
                fontWeight: 700,
                color: "var(--tj-text-primary)",
              }}
            >
              {citation.documentTitle}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="w-9 h-9 -mt-1 -mr-1 rounded-xl flex items-center justify-center text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)] hover:text-[var(--tj-text-primary)] transition-all active:scale-90 shrink-0"
          >
            <X size={18} />
          </button>
        </div>

        <div
          className="mt-2.5 flex items-center flex-wrap gap-x-3 gap-y-1"
          style={{ fontSize: 13, color: "var(--tj-text-secondary)" }}
        >
          <div className="flex items-center gap-1.5">
            <Scale size={14} className="opacity-60" />
            <span style={{ fontWeight: 600, color: "var(--tj-text-primary)" }}>
              {location}
            </span>
          </div>
          <span style={{ color: "var(--tj-text-muted)" }} className="opacity-40">·</span>
          <div className="flex items-center gap-1.5">
            <FileText size={14} className="opacity-60" />
            <span>Halaman {citation.pageNumber}</span>
          </div>
        </div>
      </header>

      {/* TOOLBAR */}
      <div className="px-4 h-12 flex items-center gap-3 border-b border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)]/40 shrink-0">
        <div className="flex items-center bg-[var(--tj-surface)]/80 rounded-xl border border-[var(--tj-border-subtle)] p-0.5 shadow-sm">
          <ToolbarBtn
            disabled={!prev}
            onClick={() => prev && onSelect(prev)}
          >
            <ChevronLeft size={16} />
          </ToolbarBtn>
          <div className="w-px h-4 bg-[var(--tj-border-subtle)]" />
          <span
            className="px-3 select-none flex items-center"
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: "var(--tj-text-secondary)",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {idx + 1} / {allCitations.length}
          </span>
          <div className="w-px h-4 bg-[var(--tj-border-subtle)]" />
          <ToolbarBtn
            disabled={!next}
            onClick={() => next && onSelect(next)}
          >
            <ChevronRight size={16} />
          </ToolbarBtn>
        </div>

        <div className="flex-1" />

        <div className="flex items-center bg-[var(--tj-surface)]/80 rounded-xl border border-[var(--tj-border-subtle)] p-0.5 shadow-sm">
          <ToolbarBtn
            onClick={() => setZoom((z) => Math.max(50, z - 10))}
            disabled={zoom <= 50}
          >
            <Minus size={14} />
          </ToolbarBtn>
          <span
            className="px-2 select-none flex items-center justify-center"
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: "var(--tj-text-secondary)",
              minWidth: 44,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {zoom}%
          </span>
          <ToolbarBtn
            onClick={() => setZoom((z) => Math.min(200, z + 10))}
            disabled={zoom >= 200}
          >
            <Plus size={14} />
          </ToolbarBtn>
        </div>

        <ToolbarBtn className="w-9 h-9 rounded-xl bg-[var(--tj-surface)] border border-[var(--tj-border-subtle)] shadow-sm">
          <Maximize2 size={15} />
        </ToolbarBtn>
      </div>

      {/* PDF VIEW */}
      <div className="flex-1 overflow-y-auto bg-[var(--tj-app-bg)]/40 tj-scroll p-6">
        <div
          className="relative mx-auto bg-white rounded-xl overflow-hidden shadow-2xl"
          style={{
            width: `${zoom}%`,
            maxWidth: "100%",
            aspectRatio: "1 / 1.414",
            transition: "width 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
          }}
        >
          <PlaceholderPdfPage citation={citation} location={location} />
        </div>
        
        {/* EXCERPT CARD */}
        <section className="mt-8 mb-6">
          <div className="flex items-center justify-between mb-3 px-1">
            <h3
              className="uppercase"
              style={{
                fontSize: 11,
                letterSpacing: "0.1em",
                fontWeight: 700,
                color: "var(--tj-text-muted)",
              }}
            >
              Kutipan Relevan
            </h3>
            <button
              onClick={copyExcerpt}
              className={`flex items-center gap-1.5 h-7 px-3 rounded-lg transition-all active:scale-95 ${copied ? "bg-[var(--tj-success)]/10 text-[var(--tj-success)]" : "bg-[var(--tj-surface)] text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)]"}`}
              style={{ fontSize: 12, fontWeight: 600 }}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
              {copied ? "Tersalin" : "Salin"}
            </button>
          </div>
          <div className="rounded-2xl bg-[var(--tj-surface)] border border-[var(--tj-border-subtle)] p-5 shadow-sm relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-1 h-full bg-[var(--tj-accent)] opacity-80" />
            <blockquote
              style={{
                fontSize: 15,
                lineHeight: "24px",
                color: "var(--tj-text-primary)",
                fontStyle: "italic",
              }}
            >
              "{citation.excerpt}"
            </blockquote>
          </div>
        </section>

        {/* DETAILS */}
        <section className="mb-8">
          <h3
            className="uppercase mb-3 px-1"
            style={{
              fontSize: 11,
              letterSpacing: "0.1em",
              fontWeight: 700,
              color: "var(--tj-text-muted)",
            }}
          >
            Informasi Dokumen
          </h3>
          <div className="rounded-2xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] overflow-hidden shadow-sm divide-y divide-[var(--tj-border-subtle)]">
            <MetaRow label="Lokasi">{location}</MetaRow>
            <MetaRow label="Halaman">Hal. {citation.pageNumber}</MetaRow>
            <MetaRow label="Yurisdiksi">Republik Indonesia</MetaRow>
            <MetaRow label="Domain">{citation.sourceDomain ?? "Tidak tersedia"}</MetaRow>
            <MetaRow label="Hash File" mono>{citation.fileHash ?? "Tidak tersedia"}</MetaRow>
          </div>
        </section>
      </div>

      {/* FOOTER */}
      <footer className="border-t border-[var(--tj-border-subtle)] p-4 bg-[var(--tj-surface)]/60 backdrop-blur-xl shrink-0">
        <div
          className="flex items-center justify-center gap-2.5 min-h-11 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)] px-3 text-center text-[var(--tj-text-secondary)]"
          style={{ fontSize: 14, fontWeight: 700 }}
        >
          PDF/BBox viewer belum disajikan oleh backend
        </div>
      </footer>
    </>
  );
}

function ToolbarBtn({
  children,
  onClick,
  disabled,
  className = "",
  ...rest
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  [k: string]: any;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-9 h-9 flex items-center justify-center text-[var(--tj-text-secondary)] hover:text-[var(--tj-text-primary)] hover:bg-[var(--tj-surface-hover)] disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-[var(--tj-text-secondary)] transition-all rounded-lg ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

function MetaRow({
  label,
  children,
  mono,
}: {
  label: string;
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-4 px-4 py-3.5">
      <span style={{ fontSize: 13, color: "var(--tj-text-muted)", fontWeight: 500 }}>{label}</span>
      <span
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: "var(--tj-text-primary)",
          fontFamily: mono ? "'JetBrains Mono', ui-monospace, monospace" : undefined,
        }}
        className="truncate flex-1 text-right"
      >
        {children}
      </span>
    </div>
  );
}

function PlaceholderPdfPage({ citation, location }: { citation: Citation; location: string }) {
  return (
    <div
      className="absolute inset-0 px-10 py-12 text-[#1a1a1a] select-none"
      style={{ fontFamily: "'Times New Roman', serif" }}
    >
      <div className="text-center mb-1 text-[8px] tracking-[0.2em] font-bold opacity-60">REPUBLIK INDONESIA</div>
      <div className="text-center mb-8 text-[10px] font-bold leading-tight">
        {citation.documentTitle.toUpperCase()}
      </div>

      <div className="space-y-2 opacity-30">
        <div className="h-[2px] w-full bg-current" />
        <div className="h-[2px] w-[95%] bg-current" />
        <div className="h-[2px] w-[98%] bg-current" />
        <div className="h-[2px] w-[88%] bg-current" />
      </div>

      <div className="mt-6 mb-3 text-[11px] font-bold">{location}</div>

      <div className="relative">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.25 }}
          className="absolute -inset-1 bg-yellow-400 rounded-sm"
        />
        <div className="space-y-2 relative">
          <div className="h-[2px] w-full bg-current" />
          <div className="h-[2px] w-full bg-current" />
          <div className="h-[2px] w-[94%] bg-current" />
        </div>
      </div>

      <div className="space-y-2 mt-6 opacity-30">
        <div className="h-[2px] w-full bg-current" />
        <div className="h-[2px] w-[97%] bg-current" />
        <div className="h-[2px] w-[92%] bg-current" />
        <div className="h-[2px] w-[99%] bg-current" />
      </div>

      <div className="absolute bottom-10 left-10 right-10 flex justify-between text-[7px] font-bold opacity-40">
        <span>PLACEHOLDER PDF/BBOX</span>
        <span>HALAMAN {citation.pageNumber}</span>
      </div>
    </div>
  );
}

function legalUnitLabel(article?: string, paragraph?: string) {
  const base = article || "UUD";
  const label = /^pasal\b/i.test(base) ? base : `Pasal ${base}`;
  return paragraph ? `${label} ayat (${paragraph})` : label;
}
