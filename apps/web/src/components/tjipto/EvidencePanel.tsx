import type { ButtonHTMLAttributes, CSSProperties, KeyboardEvent, PointerEvent, ReactNode, RefObject } from "react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
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
  Check,
  FileText,
  Bookmark,
} from "lucide-react";
import type { Citation } from "../../lib/types";
import { getLegalViewerPayload, pdfAccessUrl, saveLegalBookmark, type ViewerPayload } from "../../lib/api";
import { bboxToViewportPercent } from "../../lib/pdfBBox";
import { canvasBackingStore, fitWidthScale, isRenderCancellation, RenderTaskOwner, visiblePageWindow } from "../../lib/pdfViewer";

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_DEFAULT_WIDTH = 440;
const SIDEBAR_MAX_WIDTH = 760;
const CHAT_MIN_WIDTH = 280;
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
  const [split, setSplit] = useState(false);
  const [availableWidth, setAvailableWidth] = useState(0);
  const sidebarMaxWidth = Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, availableWidth - CHAT_MIN_WIDTH));

  useEffect(() => {
    const splitQuery = window.matchMedia("(min-width: 768px)");
    const update = () => {
      setSplit(splitQuery.matches);
    };
    update();
    splitQuery.addEventListener("change", update);
    return () => {
      splitQuery.removeEventListener("change", update);
    };
  }, []);

  useLayoutEffect(() => {
    const update = () => {
      const workspace = document.querySelector<HTMLElement>("[data-evidence-workspace]");
      const navigation = document.querySelector<HTMLElement>("[data-tjipto-navigation]");
      setAvailableWidth(Math.max(0, (workspace?.clientWidth ?? window.innerWidth) - (navigation?.getBoundingClientRect().width ?? 0)));
    };
    const observer = new ResizeObserver(update);
    const workspace = document.querySelector<HTMLElement>("[data-evidence-workspace]");
    const navigation = document.querySelector<HTMLElement>("[data-tjipto-navigation]");
    if (workspace) observer.observe(workspace);
    if (navigation) observer.observe(navigation);
    window.addEventListener("resize", update);
    update();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [citation]);

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
    setSidebarWidth(clamp(nextWidth, SIDEBAR_MIN_WIDTH, sidebarMaxWidth));
  };

  const resizeWithKeyboard = (event: KeyboardEvent<HTMLButtonElement>) => {
    const next = {
      ArrowLeft: sidebarWidth + 16,
      ArrowRight: sidebarWidth - 16,
      Home: SIDEBAR_MIN_WIDTH,
      End: sidebarMaxWidth,
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    setSidebarWidth(clamp(next, SIDEBAR_MIN_WIDTH, sidebarMaxWidth));
  };

  useEffect(() => {
    setPdfOnly(false);
    if (!citation) {
      setIsResizing(false);
    }
  }, [citation]);

  useEffect(() => {
    if (split) {
      setSidebarWidth((width) => clamp(width, SIDEBAR_MIN_WIDTH, sidebarMaxWidth));
    }
  }, [sidebarMaxWidth, split]);

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
            className={`${split ? "hidden" : ""} fixed inset-0 z-40 bg-black/40`}
          />
          <motion.aside
            id="tjipto-evidence-panel"
            key="ev-panel"
            initial={pdfOnly ? { opacity: 0 } : { opacity: 0 }}
            animate={pdfOnly ? { opacity: 1 } : { x: 0, opacity: 1 }}
            exit={pdfOnly ? { opacity: 0 } : { opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.2, 0.8, 0.2, 1] }}
            className={`${pdfOnly ? "fixed inset-0 z-50 w-full max-w-none" : split ? "static z-10 w-[var(--tj-evidence-panel-width)]" : "fixed inset-y-0 right-0 z-50 w-full sm:w-[460px] sm:max-w-[95vw]"} flex flex-col shrink-0 h-full bg-[var(--tj-surface)] border-l border-[var(--tj-glass-border)] shadow-2xl [container-type:inline-size]`}
            style={{ "--tj-evidence-panel-width": `${sidebarWidth}px` } as CSSProperties}
            data-evidence-panel={pdfOnly ? "expanded" : "normal"}
            data-evidence-mode={pdfOnly ? "expanded" : split ? "split" : "drawer"}
          >
            {!pdfOnly && split && (
              <button
                type="button"
                aria-label="Resize evidence panel"
                aria-controls="tjipto-evidence-panel"
                aria-orientation="vertical"
                aria-valuemin={SIDEBAR_MIN_WIDTH}
                aria-valuemax={sidebarMaxWidth}
                aria-valuenow={Math.round(sidebarWidth)}
                role="separator"
                title="Resize evidence panel"
                onPointerDown={startResize}
                onPointerMove={resize}
                onPointerUp={stopResize}
                onPointerCancel={stopResize}
                onKeyDown={resizeWithKeyboard}
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
  const [saved, setSaved] = useState(false);
  const [viewer, setViewer] = useState<ViewerPayload | null>(null);
  const [viewerError, setViewerError] = useState(false);
  const [renderFailed, setRenderFailed] = useState(false);
  const pdfScrollRef = useRef<HTMLDivElement | null>(null);
  const zoomAnchorRef = useRef<{ page: number; ratio: number } | null>(null);
  const location = legalUnitLabel(citation.article, citation.paragraph);
  const pageNumber = viewer?.page_numbers?.[0] ?? citation.pageNumber;
  const sourceStatus = viewer?.source_status_label ?? citation.sourceStatusLabel ?? sourceStatusLabel(citation.sourceRole, citation.temporalContext);
  const markRenderFailed = useCallback(() => {
    setRenderFailed(true);
    setViewer((current) => current ? { ...current, rendering_available: false } : current);
  }, []);

  useEffect(() => setSaved(false), [citation.publicTargetId]);

  useEffect(() => {
    let stale = false;
    setViewer(null);
    setViewerError(false);
    setRenderFailed(false);
    const request = getLegalViewerPayload(citation.publicTargetId);
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
  }, [citation.publicTargetId, citation.viewerMode]);

  const savePointer = async () => {
    try {
      const bookmark = await saveLegalBookmark(citation.publicTargetId);
      if (bookmark) setSaved(true);
    } catch {
      setSaved(false);
    }
  };

  const changeZoom = (delta: number) => {
    const scroller = pdfScrollRef.current;
    if (scroller) {
      const center = scroller.getBoundingClientRect().top + scroller.clientHeight / 2;
      const pages = [...scroller.querySelectorAll<HTMLElement>("[data-pdf-page]")];
      const page = pages.reduce<HTMLElement | null>((nearest, candidate) => {
        if (!nearest) return candidate;
        const distance = (node: HTMLElement) => {
          const rect = node.getBoundingClientRect();
          return center < rect.top ? rect.top - center : center > rect.bottom ? center - rect.bottom : 0;
        };
        return distance(candidate) < distance(nearest) ? candidate : nearest;
      }, null);
      if (page) {
        const rect = page.getBoundingClientRect();
        zoomAnchorRef.current = {
          page: Number(page.dataset.pdfPage),
          ratio: clamp((center - rect.top) / rect.height, 0, 1),
        };
      }
    }
    setZoom((value) => clamp(value + delta, 50, 200));
  };

  useLayoutEffect(() => {
    const anchor = zoomAnchorRef.current;
    const scroller = pdfScrollRef.current;
    if (!anchor || !scroller) return;
    const page = scroller.querySelector<HTMLElement>(`[data-pdf-page="${anchor.page}"]`);
    if (page) {
      const scrollerRect = scroller.getBoundingClientRect();
      const pageRect = page.getBoundingClientRect();
      const anchoredPosition = pageRect.top + pageRect.height * anchor.ratio;
      scroller.scrollTop += anchoredPosition - (scrollerRect.top + scroller.clientHeight / 2);
    }
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
            onClick={() => changeZoom(-25)}
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
            onClick={() => changeZoom(25)}
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
            className="relative mx-auto w-full min-w-0 rounded-xl border border-[var(--tj-border-subtle)] bg-[var(--tj-surface)] shadow-sm"
            style={{
              minHeight: pdfOnly ? PDF_ONLY_MIN_HEIGHT : PDF_NORMAL_MIN_HEIGHT,
            }}
          >
            {viewer?.pdf_access_available && viewer.pdf?.access_url && !renderFailed ? (
              <RenderedViewer
                key={`${citation.viewerMode ?? "evidence"}:${citation.publicTargetId}:${viewer.bbox_rectangles?.length ?? 0}`}
                viewer={viewer}
                targetPage={pageNumber}
                zoom={zoom / 100}
                scrollRoot={pdfScrollRef}
                onRenderFailed={markRenderFailed}
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
  targetPage,
  zoom,
  scrollRoot,
  onRenderFailed,
}: {
  viewer: ViewerPayload;
  targetPage: number;
  zoom: number;
  scrollRoot: RefObject<HTMLDivElement | null>;
  onRenderFailed: () => void;
}) {
  const documentRef = useRef<HTMLDivElement | null>(null);
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const intersectingPages = useRef(new Set<number>());
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [availableWidth, setAvailableWidth] = useState(0);
  const [pageRatio, setPageRatio] = useState(Math.SQRT2);
  const [visiblePages, setVisiblePages] = useState(() => new Set([targetPage]));
  const [targetRendered, setTargetRendered] = useState(false);
  const [firstRenderedPage, setFirstRenderedPage] = useState<number | null>(null);
  const pdfUrl = pdfAccessUrl(viewer);
  const markRendered = useCallback((pageNumber: number) => {
    setFirstRenderedPage((current) => current ?? pageNumber);
    if (pageNumber === targetPage) setTargetRendered(true);
  }, [targetPage]);

  useEffect(() => {
    let stale = false;
    setPdf(null);
    setTargetRendered(false);
    setFirstRenderedPage(null);
    setVisiblePages(new Set([targetPage]));
    if (!pdfUrl) return;

    const loadingTask = pdfjs.getDocument({ url: pdfUrl });
    loadingTask.promise
      .then((loaded) => {
        if (!stale) setPdf(loaded);
      })
      .catch(() => {
        if (!stale) onRenderFailed();
      });

    return () => {
      stale = true;
      void loadingTask.destroy();
    };
  }, [pdfUrl, onRenderFailed, targetPage]);

  useLayoutEffect(() => {
    const node = documentRef.current?.parentElement;
    if (!node) return;
    const update = () => setAvailableWidth(Math.max(1, node.clientWidth - 24));
    const observer = new ResizeObserver(update);
    observer.observe(node);
    update();
    return () => observer.disconnect();
  }, [pdf]);

  useEffect(() => {
    if (!pdf || !scrollRoot.current) return;
    const intersections = intersectingPages.current;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const page = Number((entry.target as HTMLElement).dataset.pdfPage);
          if (entry.isIntersecting) intersections.add(page);
          else intersections.delete(page);
        }
        const next = new Set(intersections);
        if (next.size) setVisiblePages(next);
      },
      { root: scrollRoot.current },
    );
    Object.values(pageRefs.current).forEach((node) => {
      if (node) observer.observe(node);
    });
    return () => {
      observer.disconnect();
      intersections.clear();
    };
  }, [pdf, scrollRoot]);

  useEffect(() => {
    if (!pdf) return;
    requestAnimationFrame(() => pageRefs.current[targetPage]?.scrollIntoView({ block: "center" }));
  }, [pdf, targetPage]);

  if (!pdf) {
    return (
      <div className="min-h-[260px] flex items-center justify-center text-sm text-[var(--tj-text-secondary)]">
        Memuat dokumen PDF
      </div>
    );
  }

  const target = clamp(targetPage, 1, pdf.numPages);
  const renderPages = targetRendered ? visiblePageWindow(visiblePages, pdf.numPages) : new Set([target]);
  const logicalWidth = Math.max(1, availableWidth * zoom);
  return (
    <div
      ref={documentRef}
      className="w-max min-w-full bg-white px-3 py-4 space-y-4"
      data-pdf-document="windowed"
      data-page-count={pdf.numPages}
      data-active-canvas-count={renderPages.size}
      data-first-rendered-page={firstRenderedPage ?? ""}
    >
      {Array.from({ length: pdf.numPages }, (_, index) => {
        const pageNumber = index + 1;
        const shouldRender = renderPages.has(pageNumber);
        return (
          <div
            key={pageNumber}
            ref={(node) => {
              pageRefs.current[pageNumber] = node;
            }}
            data-pdf-page={pageNumber}
            data-canvas-active={shouldRender ? "true" : "false"}
            className="relative mx-auto overflow-hidden bg-white shadow-sm"
            style={{ width: logicalWidth, height: logicalWidth * pageRatio }}
          >
            {shouldRender && <PdfPage
              pdf={pdf}
              pageNumber={pageNumber}
              active={pageNumber === target}
              boxes={viewer.bbox_rectangles ?? []}
              availableWidth={availableWidth}
              zoom={zoom}
              onRenderFailed={onRenderFailed}
              onRendered={markRendered}
              onPageRatio={setPageRatio}
            />}
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
  availableWidth,
  zoom,
  onRenderFailed,
  onRendered,
  onPageRatio,
}: {
  pdf: PDFDocumentProxy;
  pageNumber: number;
  active: boolean;
  boxes: NonNullable<ViewerPayload["bbox_rectangles"]>;
  availableWidth: number;
  zoom: number;
  onRenderFailed: () => void;
  onRendered: (pageNumber: number) => void;
  onPageRatio: (ratio: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const renderTaskOwner = useRef(new RenderTaskOwner<ReturnType<PDFPageProxy["render"]>>());
  const [pageViewport, setPageViewport] = useState<ReturnType<PDFPageProxy["getViewport"]> | null>(null);
  const [rendered, setRendered] = useState(false);
  const pageBoxes = boxes.filter(
    (box) => box.page_number === pageNumber && box.viewer_highlightable === true,
  );

  useEffect(() => {
    let stale = false;
    let currentTask: ReturnType<PDFPageProxy["render"]> | null = null;
    const taskOwner = renderTaskOwner.current;
    const canvas = canvasRef.current;
    if (!canvas || availableWidth <= 0) return;
    setRendered(false);
    taskOwner.cancel();

    pdf.getPage(pageNumber)
      .then((page) => {
        if (stale) return null;
        const baseViewport = page.getViewport({ scale: 1 });
        const viewport = page.getViewport({ scale: fitWidthScale(baseViewport.width, availableWidth, zoom) });
        const backingStore = canvasBackingStore(viewport.width, viewport.height, window.devicePixelRatio);
        const context = canvas.getContext("2d");
        if (!context) throw new Error("canvas_unavailable");
        canvas.width = backingStore.width;
        canvas.height = backingStore.height;
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        setPageViewport(viewport);
        onPageRatio(baseViewport.height / baseViewport.width);
        const task = page.render({
          canvas,
          canvasContext: context,
          viewport,
          transform: backingStore.ratio === 1 ? undefined : [backingStore.ratio, 0, 0, backingStore.ratio, 0, 0],
        });
        currentTask = task;
        taskOwner.replace(task);
        return task.promise;
      })
      .then(() => {
        if (!stale && currentTask && taskOwner.isCurrent(currentTask)) {
          setRendered(true);
          taskOwner.finish(currentTask);
          onRendered(pageNumber);
        }
      })
      .catch((error: unknown) => {
        if (!stale && !isRenderCancellation(error)) onRenderFailed();
      });

    return () => {
      stale = true;
      taskOwner.cancel();
    };
  }, [availableWidth, onPageRatio, onRenderFailed, onRendered, pageNumber, pdf, zoom]);

  return (
    <>
      <canvas
        ref={canvasRef}
        aria-label={`Halaman sumber ${pageNumber}`}
        data-rendered={rendered ? "true" : "false"}
        className="block"
      />
      {!rendered && (
        <div className="absolute inset-0 min-h-[260px] flex items-center justify-center text-sm text-[var(--tj-text-secondary)]">
          Memuat halaman PDF
        </div>
      )}
      {pageViewport && rendered && <div className="absolute inset-0 pointer-events-none">
        {pageBoxes.map((box) => {
          const rect = bboxToViewportPercent(box, pageViewport);
          if (!rect.ok) return null;
          return (
            <span
              key={box.public_rectangle_id ?? `${box.page_number}:${box.x0}:${box.y0}:${box.x1}:${box.y1}`}
              data-bbox-highlight={active ? "active" : "related"}
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
