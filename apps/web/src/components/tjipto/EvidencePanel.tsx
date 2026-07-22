import type { ButtonHTMLAttributes, ClipboardEvent, CSSProperties, PointerEvent, ReactNode } from "react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import * as pdfjs from "pdfjs-dist";
import type { PDFDocumentProxy, PDFPageProxy } from "pdfjs-dist/types/src/display/api";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
import {
  X,
  Minus,
  Plus,
  Maximize2,
  Minimize2,
  Copy,
  Check,
  FileText,
  Bookmark,
} from "lucide-react";
import type { Citation } from "../../lib/types";
import { getDocumentViewerPayload, getLegalViewerPayload, pdfAccessUrl, saveLegalBookmark, type ViewerPayload } from "../../lib/api";
import { bboxToViewportPercent } from "../../lib/pdfBBox";

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const SIDEBAR_MIN_WIDTH = 360;
const SIDEBAR_DEFAULT_WIDTH = 440;
const SIDEBAR_MAX_WIDTH = 760;
const PDF_NORMAL_MIN_HEIGHT = 260;
const PDF_ONLY_MIN_HEIGHT = "calc(100vh - 112px)";
const PDF_AREA_PADDING_CLASS = "p-4 sm:p-6";

interface EvidencePanelProps {
  citation: Citation | null;
  allCitations: Citation[];
  onClose: () => void;
  onSelect: (c: Citation) => void;
}

export function EvidencePanel({
  citation,
  onClose,
}: EvidencePanelProps) {
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT_WIDTH);
  const [isResizing, setIsResizing] = useState(false);
  const [pdfOnly, setPdfOnly] = useState(false);

  const startResize = (event: PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setIsResizing(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const stopResize = (event: PointerEvent<HTMLButtonElement>) => {
    setIsResizing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const resize = (event: PointerEvent<HTMLButtonElement>) => {
    if (!isResizing) return;
    const nextWidth = window.innerWidth - event.clientX;
    setSidebarWidth(clamp(nextWidth, SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, window.innerWidth - 96)));
  };

  useEffect(() => {
    setPdfOnly(false);
    if (!citation) {
      setIsResizing(false);
    }
  }, [citation]);

  return (
    <AnimatePresence>
      {citation && (
        <>
          <motion.div
            key="ev-overlay-bg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={onClose}
            className="lg:hidden fixed inset-0 z-40 bg-black/40"
          />
          <motion.aside
            key="ev-panel"
            initial={pdfOnly ? { opacity: 0 } : { opacity: 0 }}
            animate={pdfOnly ? { opacity: 1 } : { x: 0, opacity: 1 }}
            exit={pdfOnly ? { opacity: 0 } : { opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.2, 0.8, 0.2, 1] }}
            className={`${pdfOnly ? "fixed inset-0 z-50 w-full max-w-none" : "fixed lg:static inset-y-0 right-0 z-50 lg:z-10 w-full sm:w-[460px] md:w-[var(--tj-evidence-panel-width)] sm:max-w-[95vw]"} flex flex-col shrink-0 h-full bg-[var(--tj-surface)]/80 lg:bg-[var(--tj-surface)]/60 backdrop-blur-3xl border-l border-[var(--tj-glass-border)] shadow-2xl`}
            style={{ "--tj-evidence-panel-width": `${sidebarWidth}px` } as CSSProperties}
            data-evidence-panel={pdfOnly ? "expanded" : "normal"}
          >
            {!pdfOnly && (
              <button
                type="button"
                aria-label="Resize evidence panel"
                title="Resize evidence panel"
                onPointerDown={startResize}
                onPointerMove={resize}
                onPointerUp={stopResize}
                onPointerCancel={stopResize}
                className={`hidden md:block absolute inset-y-0 -left-1.5 z-20 w-3 cursor-col-resize touch-none transition-colors ${isResizing ? "bg-[var(--tj-accent-soft)]" : "hover:bg-[var(--tj-accent-soft)]"}`}
                data-evidence-resize-handle="true"
              />
            )}
            <EvidenceContent
              citation={citation}
              onClose={onClose}
              pdfOnly={pdfOnly}
              onTogglePdfOnly={() => setPdfOnly((value) => !value)}
            />
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function EvidenceContent({
  citation,
  onClose,
  pdfOnly,
  onTogglePdfOnly,
}: {
  citation: Citation;
  onClose: () => void;
  pdfOnly: boolean;
  onTogglePdfOnly: () => void;
}) {
  const documentMode = citation.viewerMode === "document";
  const [zoom, setZoom] = useState(100);
  const [copied, setCopied] = useState(false);
  const [saved, setSaved] = useState(false);
  const [viewer, setViewer] = useState<ViewerPayload | null>(null);
  const [viewerError, setViewerError] = useState(false);
  const [renderFailed, setRenderFailed] = useState(false);
  const pdfScrollRef = useRef<HTMLDivElement | null>(null);
  const zoomAnchorRef = useRef<{ top: number; scrollable: number } | null>(null);
  const location = legalUnitLabel(citation.article, citation.paragraph);
  const pageNumber = viewer?.page_numbers?.[0] ?? citation.pageNumber;
  const sourceStatus = viewer?.source_status_label ?? citation.sourceStatusLabel ?? sourceStatusLabel(citation.sourceRole, citation.temporalContext);

  useEffect(() => setSaved(false), [citation.documentId]);

  useEffect(() => {
    let stale = false;
    setViewer(null);
    setViewerError(false);
    setRenderFailed(false);
    const request = citation.viewerMode === "document" && citation.sourceDocumentId
      ? getDocumentViewerPayload(citation.sourceDocumentId)
      : getLegalViewerPayload(citation.documentId, citation.relationId);
    request
      .then((payload) => {
        if (!stale) setViewer(payload);
      })
      .catch(() => {
        if (!stale) setViewerError(true);
      });
    return () => {
      stale = true;
    };
  }, [citation.documentId, citation.viewerMode, citation.relationId]);

  const copyExcerpt = async () => {
    const text = (citation.copyText ?? citation.excerpt).replace(/\r\n?/g, "\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const normalizeSelectionCopy = (event: ClipboardEvent<HTMLElement>) => {
    const selected = window.getSelection()?.toString() ?? "";
    if (!selected.trim()) return;
    event.preventDefault();
    event.clipboardData.clearData();
    event.clipboardData.setData("text/plain", selected.replace(/\r\n?/g, "\n").split("\n").map((line) => line.trimStart()).join("\n"));
  };

  const savePointer = async () => {
    try {
      const bookmark = await saveLegalBookmark(citation.documentId);
      if (bookmark) setSaved(true);
    } catch {
      setSaved(false);
    }
  };

  const changeZoom = (delta: number) => {
    const scroller = pdfScrollRef.current;
    zoomAnchorRef.current = scroller
      ? { top: scroller.scrollTop, scrollable: scroller.scrollHeight - scroller.clientHeight }
      : null;
    setZoom((value) => clamp(value + delta, 50, 200));
  };

  useLayoutEffect(() => {
    const anchor = zoomAnchorRef.current;
    const scroller = pdfScrollRef.current;
    if (!anchor || !scroller) return;
    const nextScrollable = scroller.scrollHeight - scroller.clientHeight;
    scroller.scrollTop = anchor.scrollable > 0
      ? (anchor.top / anchor.scrollable) * nextScrollable
      : anchor.top;
    zoomAnchorRef.current = null;
  }, [zoom]);

  return (
    <>
      {/* HEADER */}
      {!pdfOnly && <header className="px-6 pt-5 pb-4 border-b border-[var(--tj-border-subtle)] shrink-0">
        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1.5 min-w-0">
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

      </header>}

      {/* TOOLBAR */}
      <div className="px-4 sm:px-6 h-12 flex items-center gap-3 border-b border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)]/40 shrink-0">
        <div className="flex items-center bg-[var(--tj-surface)]/80 rounded-xl border border-[var(--tj-border-subtle)] p-0.5 shadow-sm">
          <ToolbarBtn
            onClick={() => changeZoom(-10)}
            disabled={zoom <= 50}
            aria-label="Zoom out"
            title="Zoom out"
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
            onClick={() => changeZoom(10)}
            disabled={zoom >= 200}
            aria-label="Zoom in"
            title="Zoom in"
          >
            <Plus size={14} />
          </ToolbarBtn>
        </div>

        <div className="flex-1" />

        <ToolbarBtn
          onClick={onTogglePdfOnly}
          className="w-9 h-9 rounded-xl bg-[var(--tj-surface)] border border-[var(--tj-border-subtle)] shadow-sm"
          aria-label={pdfOnly ? "Exit PDF-only mode" : "Expand PDF-only mode"}
          title={pdfOnly ? "Exit PDF-only mode" : "Expand PDF-only mode"}
        >
          {pdfOnly ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
        </ToolbarBtn>
        {!pdfOnly && !documentMode && <ToolbarBtn
          onClick={savePointer}
          className="w-9 h-9 rounded-xl bg-[var(--tj-surface)] border border-[var(--tj-border-subtle)] shadow-sm"
          aria-label="Simpan bookmark sementara"
          title="Simpan bookmark sementara"
        >
          {saved ? <Check size={15} /> : <Bookmark size={15} />}
        </ToolbarBtn>}
      </div>

      <div className="flex-1 min-h-0 flex flex-col bg-[var(--tj-app-bg)]/40">
        {/* PDF VIEW */}
        <div
          ref={pdfScrollRef}
          className={`min-h-0 overflow-auto tj-scroll ${PDF_AREA_PADDING_CLASS} ${pdfOnly || documentMode ? "flex-1" : "basis-[54%] border-b border-[var(--tj-border-subtle)]"}`}
          data-evidence-pdf-area={pdfOnly ? "expanded" : documentMode ? "document" : "normal"}
        >
          <div
            className={`relative mx-auto rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] overflow-hidden shadow-sm ${pdfOnly ? "max-w-[min(1180px,100%)]" : ""}`}
            style={{
              width: `${zoom}%`,
              maxWidth: zoom <= 100 ? "100%" : "none",
              minHeight: pdfOnly ? PDF_ONLY_MIN_HEIGHT : PDF_NORMAL_MIN_HEIGHT,
              transition: "width 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
            }}
          >
            {viewer?.pdf_access_available && viewer.pdf?.access_url && !renderFailed ? (
              <RenderedViewer
                key={`${citation.viewerMode ?? "evidence"}:${viewer.source_document_id}:${viewer.evidence_id ?? "document"}:${viewer.bbox_rectangles?.length ?? 0}`}
                viewer={viewer}
                citation={citation}
                targetPage={pageNumber}
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
        </div>

        {!pdfOnly && !documentMode && <div className="flex-1 min-h-0 overflow-y-auto tj-scroll px-6 py-6" data-evidence-detail-area="normal">
          {/* EXCERPT CARD */}
          <section className="mb-6">
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
                {citation.panelSection ?? "Kutipan Relevan"}
              </h3>
              <button
                type="button"
                aria-label="Salin kutipan relevan"
                onClick={copyExcerpt}
                className={`flex items-center gap-1.5 h-7 px-3 rounded-lg transition-all active:scale-95 ${copied ? "bg-[var(--tj-success)]/10 text-[var(--tj-success)]" : "bg-[var(--tj-surface)] text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)]"}`}
                style={{ fontSize: 12, fontWeight: 600 }}
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
                {copied ? "Tersalin" : "Salin"}
              </button>
            </div>
            <p className="sr-only" aria-live="polite">{copied ? "Kutipan relevan tersalin sebagai teks biasa." : ""}</p>
            <div className="rounded-2xl bg-[var(--tj-surface)] border border-[var(--tj-border-subtle)] p-5 shadow-sm relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-1 h-full bg-[var(--tj-accent)] opacity-80" />
              <blockquote
                onCopy={normalizeSelectionCopy}
                tabIndex={0}
                style={{
                  fontSize: 15,
                  lineHeight: "24px",
                  color: "var(--tj-text-primary)",
                  whiteSpace: "pre-wrap",
                }}
              >
                {(citation.layoutLines ?? [{ text: citation.displayText ?? citation.excerpt, line_order: 0, paragraph_id: "support", alignment: "unknown", indent: 0, source_bbox_refs: [] }]).map((line) => (
                  <span
                    key={`${line.paragraph_id}:${line.line_order}`}
                    style={{ display: "block", textAlign: line.alignment === "unknown" ? "left" : line.alignment, paddingLeft: line.indent ? `${line.indent}px` : undefined }}
                  >
                    {line.text}
                  </span>
                ))}
              </blockquote>
            </div>
          </section>

          {/* DETAILS */}
          <section className="mb-6">
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
              <MetaRow label="Halaman">{pageNumber}</MetaRow>
              <MetaRow label="Status Sumber">{sourceStatus}</MetaRow>
              <MetaRow label="Yurisdiksi">Republik Indonesia</MetaRow>
              <MetaRow label="Domain">{citation.sourceDomain ?? "Tidak tersedia"}</MetaRow>
            </div>
          </section>

          {/* FOOTER */}
          <footer className="pt-2">
            <div
              className="flex items-center justify-center gap-2.5 min-h-11 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface-subtle)] px-3 text-center text-[var(--tj-text-secondary)]"
              style={{ fontSize: 14, fontWeight: 700 }}
            >
              {viewer?.pdf_access_available && !renderFailed
                ? "PDF asli dirender di frontend melalui akses backend tervalidasi"
                : "PDF/BBox viewer belum tersedia untuk evidence ini"}
            </div>
          </footer>
        </div>}
      </div>
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

function RenderedViewer({
  viewer,
  citation,
  targetPage,
  onRenderFailed,
}: {
  viewer: ViewerPayload;
  citation: Citation;
  targetPage: number;
  onRenderFailed: () => void;
}) {
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [renderedPages, setRenderedPages] = useState(0);
  const pdfUrl = pdfAccessUrl(viewer);

  useEffect(() => {
    let cancelled = false;
    setPdf(null);
    setRenderedPages(0);
    if (!pdfUrl) return;

    pdfjs.getDocument({ url: pdfUrl }).promise
      .then((document) => {
        if (!cancelled) setPdf(document);
      })
      .catch(() => {
        if (!cancelled) onRenderFailed();
      });

    return () => {
      cancelled = true;
    };
  }, [pdfUrl]);

  useEffect(() => {
    if (!pdf || renderedPages < 1) return;
    pageRefs.current[targetPage]?.scrollIntoView({ block: "center" });
  }, [pdf, renderedPages, targetPage]);

  if (!pdf) {
    return (
      <div className="min-h-[260px] flex items-center justify-center text-sm text-[var(--tj-text-secondary)]">
        Memuat dokumen PDF
      </div>
    );
  }

  return (
    <div className="bg-white px-3 py-4 space-y-4" data-pdf-document="full" data-page-count={pdf.numPages}>
      {Array.from({ length: pdf.numPages }, (_, index) => {
        const pageNumber = index + 1;
        return (
          <div
            key={pageNumber}
            ref={(node) => {
              pageRefs.current[pageNumber] = node;
            }}
            data-pdf-page={pageNumber}
            className="relative mx-auto overflow-hidden bg-white shadow-sm"
            style={{ width: "100%" }}
          >
            <PdfPage
              pdf={pdf}
              pageNumber={pageNumber}
              active={pageNumber === targetPage}
              boxes={viewer.bbox_rectangles ?? []}
              citation={citation}
              onRenderFailed={onRenderFailed}
              onRendered={() => setRenderedPages((count) => count + 1)}
            />
          </div>
        );
      })}
    </div>
  );
}

function PdfPage({
  pdf,
  pageNumber,
  active,
  boxes,
  citation,
  onRenderFailed,
  onRendered,
}: {
  pdf: PDFDocumentProxy;
  pageNumber: number;
  active: boolean;
  boxes: NonNullable<ViewerPayload["bbox_rectangles"]>;
  citation: Citation;
  onRenderFailed: () => void;
  onRendered: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [pageSize, setPageSize] = useState<{ width: number; height: number } | null>(null);
  const [pageViewport, setPageViewport] = useState<ReturnType<PDFPageProxy["getViewport"]> | null>(null);
  const [rendered, setRendered] = useState(false);
  const relationProofIds = citation.relationProof ? citation.relationSourceProofBBoxRefs : undefined;
  const targetIds = new Set(citation.relationTargetBBoxRefs ?? []);
  const pageBoxes = boxes.filter(
    (box) => box.page_number === pageNumber && box.viewer_highlightable === true,
  ).filter((box) => !relationProofIds || relationProofIds.includes(String(box.bbox_id)));

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    setRendered(false);

    pdf.getPage(pageNumber)
      .then((page) => {
        if (cancelled) return;
        const viewport = page.getViewport({ scale: 1.35 });
        const context = canvas.getContext("2d");
        if (!context) throw new Error("canvas_unavailable");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const pageViewport = page.getViewport({ scale: 1 });
        setPageSize({ width: pageViewport.width, height: pageViewport.height });
        setPageViewport(pageViewport);
        return renderPdfPage(page, canvas, context, viewport);
      })
      .then(() => {
        if (!cancelled) {
          setRendered(true);
          onRendered();
        }
      })
      .catch(() => {
        if (!cancelled) onRenderFailed();
      });

    return () => {
      cancelled = true;
    };
  }, [pdf, pageNumber]);

  return (
    <>
      <canvas
        ref={canvasRef}
        aria-label={`Halaman sumber ${pageNumber}`}
        data-rendered={rendered ? "true" : "false"}
        className="block w-full h-auto"
      />
      {!rendered && (
        <div className="absolute inset-0 min-h-[260px] flex items-center justify-center text-sm text-[var(--tj-text-secondary)]">
          Memuat halaman PDF
        </div>
      )}
      {pageSize && pageViewport && rendered && <div className="absolute inset-0 pointer-events-none">
        {pageBoxes.map((box) => {
          const rect = bboxToViewportPercent(box, pageViewport);
          if (!rect.ok) return null;
          return (
            <span
              key={box.bbox_id}
              data-bbox-highlight={targetIds.has(String(box.bbox_id)) ? "target" : active ? "active" : "related"}
              data-relation-layer={relationProofIds ? (targetIds.has(String(box.bbox_id)) ? "target-emphasis" : "source-proof") : undefined}
              className="absolute"
              style={{
                left: `${rect.left}%`,
                top: `${rect.top}%`,
                width: `${rect.width}%`,
                height: `${rect.height}%`,
                background: "rgba(255, 235, 59, 0.28)",
                border: "1px solid rgba(245, 211, 39, 0.42)",
                mixBlendMode: "multiply",
              }}
            />
          );
        })}
      </div>}
    </>
  );
}

function renderPdfPage(
  page: PDFPageProxy,
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  viewport: ReturnType<PDFPageProxy["getViewport"]>,
) {
  return page.render({ canvas, canvasContext: context, viewport }).promise;
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

function sourceStatusLabel(sourceRole?: string, temporalContext?: string) {
  const role = sourceRole ?? temporalContext;
  if (role === "current_consolidated") return "Berlaku (konsolidasi saat ini)";
  if (role?.startsWith("amendment_")) return "Historis (sumber perubahan)";
  if (role === "original_historical") return "Historis (naskah asli)";
  return "Status sumber tidak tersedia";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}
