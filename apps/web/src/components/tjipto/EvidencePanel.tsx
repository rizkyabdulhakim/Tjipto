import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import * as pdfjs from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
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
  Bookmark,
} from "lucide-react";
import type { Citation } from "../../lib/types";
import { getLegalViewerPayload, pdfAccessUrl, saveLegalBookmark, type ViewerPayload } from "../../lib/api";

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

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
  const [saved, setSaved] = useState(false);
  const [viewer, setViewer] = useState<ViewerPayload | null>(null);
  const [viewerError, setViewerError] = useState(false);
  const [renderFailed, setRenderFailed] = useState(false);
  const location = legalUnitLabel(citation.article, citation.paragraph);
  const pageNumber = viewer?.page_numbers?.[0] ?? citation.pageNumber;
  const sourceHash = viewer?.source_sha256
    ? `sha256:${viewer.source_sha256}`
    : citation.fileHash;

  useEffect(() => setSaved(false), [citation.documentId]);

  useEffect(() => {
    let stale = false;
    setViewer(null);
    setViewerError(false);
    setRenderFailed(false);
    getLegalViewerPayload(citation.documentId)
      .then((payload) => {
        if (!stale) setViewer(payload);
      })
      .catch(() => {
        if (!stale) setViewerError(true);
      });
    return () => {
      stale = true;
    };
  }, [citation.documentId]);

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

  const savePointer = async () => {
    try {
      const bookmark = await saveLegalBookmark(citation.documentId);
      if (bookmark) setSaved(true);
    } catch {
      setSaved(false);
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
            <span>Halaman {pageNumber}</span>
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
        <ToolbarBtn
          onClick={savePointer}
          className="w-9 h-9 rounded-xl bg-[var(--tj-surface)] border border-[var(--tj-border-subtle)] shadow-sm"
          aria-label="Simpan bookmark sementara"
          title="Simpan bookmark sementara"
        >
          {saved ? <Check size={15} /> : <Bookmark size={15} />}
        </ToolbarBtn>
      </div>

      {/* PDF VIEW */}
      <div className="flex-1 overflow-y-auto bg-[var(--tj-app-bg)]/40 tj-scroll p-6">
        <div
          className="relative mx-auto rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] overflow-hidden shadow-sm"
          style={{
            width: `${zoom}%`,
            maxWidth: "100%",
            minHeight: 260,
            transition: "width 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
          }}
        >
          {viewer?.pdf_access_available && viewer.pdf?.access_url && !renderFailed ? (
            <RenderedViewer
              viewer={viewer}
              onRenderFailed={() => {
                setRenderFailed(true);
                setViewer((current) => current ? { ...current, rendering_available: false, render_status: "render_failed_safe" } : current);
              }}
            />
          ) : (
            <UnavailableViewer
              location={location}
              pageNumber={pageNumber}
              viewer={viewer}
              viewerError={viewerError}
            />
          )}
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
            <MetaRow label="Halaman">Hal. {pageNumber}</MetaRow>
            <MetaRow label="Yurisdiksi">Republik Indonesia</MetaRow>
            <MetaRow label="Domain">{citation.sourceDomain ?? "Tidak tersedia"}</MetaRow>
            <MetaRow label="Hash File" mono>{sourceHash ?? "Tidak tersedia"}</MetaRow>
          </div>
        </section>
      </div>

      {/* FOOTER */}
      <footer className="border-t border-[var(--tj-border-subtle)] p-4 bg-[var(--tj-surface)]/60 backdrop-blur-xl shrink-0">
        <div
          className="flex items-center justify-center gap-2.5 min-h-11 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)] px-3 text-center text-[var(--tj-text-secondary)]"
          style={{ fontSize: 14, fontWeight: 700 }}
        >
          {viewer?.pdf_access_available && !renderFailed
            ? "PDF asli dirender di frontend melalui akses backend tervalidasi"
            : "PDF/BBox viewer belum tersedia untuk evidence ini"}
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
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
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
  children: ReactNode;
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

function RenderedViewer({ viewer, onRenderFailed }: { viewer: ViewerPayload; onRenderFailed: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [pageSize, setPageSize] = useState<{ width: number; height: number } | null>(null);
  const [rendered, setRendered] = useState(false);
  const boxes = (viewer.bbox_rectangles ?? []).filter(
    (box) => box.page_number === viewer.page_number,
  );

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    const pdfUrl = pdfAccessUrl(viewer);
    if (!canvas || !pdfUrl || !viewer.page_number) return;
    setRendered(false);

    pdfjs.getDocument({ url: pdfUrl }).promise
      .then((pdf) => pdf.getPage(viewer.page_number ?? 1))
      .then((page) => {
        if (cancelled) return;
        const viewport = page.getViewport({ scale: 1.5 });
        const context = canvas.getContext("2d");
        if (!context) throw new Error("canvas_unavailable");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const pageViewport = page.getViewport({ scale: 1 });
        setPageSize({ width: pageViewport.width, height: pageViewport.height });
        return page.render({ canvas, canvasContext: context, viewport }).promise;
      })
      .then(() => {
        if (!cancelled) setRendered(true);
      })
      .catch(() => {
        if (!cancelled) onRenderFailed();
      });

    return () => {
      cancelled = true;
    };
  }, [viewer.page_number, viewer.pdf?.access_url]);

  return (
    <div className="relative bg-white">
      <canvas
        ref={canvasRef}
        aria-label={`Halaman sumber ${viewer.page_number ?? ""}`}
        data-rendered={rendered ? "true" : "false"}
        className="block w-full h-auto"
      />
      {!rendered && (
        <div className="absolute inset-0 min-h-[260px] flex items-center justify-center text-sm text-[var(--tj-text-secondary)]">
          Memuat halaman PDF
        </div>
      )}
      {pageSize && rendered && <div className="absolute inset-0 pointer-events-none">
        {boxes.map((box) => (
          <span
            key={box.bbox_id}
            className="absolute border-2 border-[var(--tj-accent)] bg-[var(--tj-pdf-highlight)] shadow-[0_0_0_1px_rgba(255,255,255,0.75)]"
            style={{
              left: `${percent(box.x0, pageSize.width)}%`,
              top: `${percent(box.y0, pageSize.height)}%`,
              width: `${percent((box.x1 ?? 0) - (box.x0 ?? 0), pageSize.width)}%`,
              height: `${percent((box.y1 ?? 0) - (box.y0 ?? 0), pageSize.height)}%`,
            }}
          />
        ))}
      </div>}
    </div>
  );
}

function percent(value: number | undefined, total: number) {
  return total > 0 ? 100 * (value ?? 0) / total : 0;
}

function UnavailableViewer({
  location,
  pageNumber,
  viewer,
  viewerError,
}: {
  location: string;
  pageNumber: number;
  viewer: ViewerPayload | null;
  viewerError: boolean;
}) {
  const loading = !viewer && !viewerError;
  const backendReady = viewer?.status === "viewer_payload_ready";
  return (
    <div
      className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 py-10 text-center"
    >
      <FileText size={28} className="text-[var(--tj-text-muted)]" />
      <div style={{ fontSize: 14, fontWeight: 700, color: "var(--tj-text-primary)" }}>
        {loading ? "Memuat metadata viewer" : "Rendering PDF/BBox belum tersedia"}
      </div>
      <div style={{ fontSize: 13, color: "var(--tj-text-secondary)", lineHeight: "20px" }}>
        {backendReady
          ? `Evidence backend tersedia untuk ${location} pada halaman ${pageNumber}, tetapi runtime menyatakan rendering_available=false sehingga panel ini tidak menampilkan halaman PDF atau overlay BBox.`
          : "Viewer runtime belum mengembalikan payload siap render untuk evidence ini."}
      </div>
    </div>
  );
}

function legalUnitLabel(article?: string, paragraph?: string) {
  const base = article || "UUD";
  const knownLabel = /^(pasal|bab|aturan|pembukaan)\b/i.test(base) || base.includes(" / ");
  const label = knownLabel ? base : `Pasal ${base}`;
  return paragraph ? `${label} ayat (${paragraph})` : label;
}
