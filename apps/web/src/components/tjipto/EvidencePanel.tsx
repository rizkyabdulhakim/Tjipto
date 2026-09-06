import type { CSSProperties, KeyboardEvent, PointerEvent } from "react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { X } from "lucide-react";
import type { Citation } from "../../lib/types";
import { getLegalViewerPayload, legalReferenceLabel, saveLegalBookmark, type ViewerPayload } from "../../lib/api";
import { LegalStatusCard } from "./LegalStatusCard";
import { PanelToolbar } from "./PanelToolbar";
import { documentRole, legalStatus } from "../../lib/legalPresentation";
import { DocumentLegalDetails } from "./DocumentLegalDetails";
import { RenderedViewer, UnavailableViewer } from "./PdfDocumentViewer";
import { clamp } from "../../lib/pdfViewer";


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
            className={`${pdfOnly ? "fixed inset-0 z-50 w-full max-w-none" : split ? "static z-10 w-[var(--tj-evidence-panel-width)]" : "fixed inset-y-0 right-0 z-50 w-full"} flex flex-col shrink-0 h-full bg-[var(--tj-surface)] border-l border-[var(--tj-glass-border)] shadow-2xl [container-type:inline-size]`}
            style={{ "--tj-evidence-panel-width": `${sidebarWidth}px` } as CSSProperties}
            data-evidence-panel={pdfOnly ? "expanded" : "normal"}
            data-evidence-mode={pdfOnly ? "expanded" : split ? "split" : "drawer"}
          >
            {!pdfOnly && split && (
              <button
                type="button"
                aria-label="Ubah lebar panel sumber"
                aria-controls="tjipto-evidence-panel"
                aria-orientation="vertical"
                aria-valuemin={SIDEBAR_MIN_WIDTH}
                aria-valuemax={sidebarMaxWidth}
                aria-valuenow={Math.round(sidebarWidth)}
                role="separator"
                title="Ubah lebar panel sumber"
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
  const documentMode = citation.viewerMode === "document" || citation.viewerMode === "catalog";
  const [zoom, setZoom] = useState(100);
  const [saved, setSaved] = useState(false);
  const [viewer, setViewer] = useState<ViewerPayload | null>(null);
  const [viewerError, setViewerError] = useState(false);
  const [renderFailed, setRenderFailed] = useState(false);
  const pdfScrollRef = useRef<HTMLDivElement | null>(null);
  const zoomAnchorRef = useRef<{ page: number; ratio: number } | null>(null);
  const location = legalReferenceLabel(citation.article, citation.paragraph);
  const pageNumber = viewer?.page_numbers?.[0] ?? citation.pageNumber;
  const sourceStatus = legalStatus(viewer?.legal_status ?? citation.legalStatus);
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
    const request = getLegalViewerPayload(citation.publicTargetId, citation.viewerMode === "catalog");
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
      {!pdfOnly && <header className="bg-[var(--tj-surface)] px-6 pt-5 pb-4 border-b border-[var(--tj-border-subtle)] shrink-0">
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
            aria-label="Tutup panel"
            title="Tutup panel"
            className="w-9 h-9 -mt-1 -mr-1 rounded-xl flex items-center justify-center text-[var(--tj-text-secondary)] hover:bg-[var(--tj-surface-hover)] hover:text-[var(--tj-text-primary)] transition-all active:scale-90 shrink-0"
          >
            <X size={18} />
          </button>
        </div>

      </header>}

      <PanelToolbar
        zoom={zoom}
        expanded={pdfOnly}
        saved={saved}
        canBookmark
        onZoom={changeZoom}
        onToggleExpanded={onTogglePdfOnly}
        onBookmark={savePointer}
      />

      <div className="flex-1 min-h-0 flex flex-col bg-[var(--tj-app-bg)]/40">
        {/* PDF VIEW */}
        <div
          ref={pdfScrollRef}
          className={`tj-pdf-scroll min-h-0 overflow-auto tj-scroll ${PDF_AREA_PADDING_CLASS} ${pdfOnly ? "flex-1" : "basis-[54%]"}`}
          data-evidence-pdf-area={pdfOnly ? "expanded" : documentMode ? "document" : "normal"}
        >
          <div
            className="tj-pdf-frame relative mx-auto w-full min-w-0 overflow-hidden rounded-2xl bg-[#121212] shadow-sm"
            data-pdf-card
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

        {!pdfOnly && (
          <div aria-hidden="true" className="flex h-4 shrink-0 items-center bg-[var(--tj-app-bg)]/40 px-6" data-evidence-section-divider>
            <div className="h-px w-full bg-[var(--tj-border-subtle)]" />
          </div>
        )}

        {!pdfOnly && <div className={`tj-info-scroll flex-1 min-h-0 overflow-y-auto tj-scroll ${PDF_AREA_PADDING_CLASS}`} data-evidence-detail-area="normal">
          <LegalStatusCard
            status={sourceStatus}
            role={documentRole(viewer?.document_role ?? citation.documentRole)}
            title={viewer?.title ?? citation.documentTitle}
            documentType={viewer?.document_type ?? citation.documentType}
            number={viewer?.number}
            year={viewer?.year}
            issuer={viewer?.issuer ?? citation.issuer}
            establishmentPlace={viewer?.establishment_place}
            signatories={viewer?.signatories}
            establishmentDate={viewer?.establishment_date ?? citation.establishmentDate}
            promulgationDate={viewer?.promulgation_date ?? citation.promulgationDate}
            effectiveDate={viewer?.effective_date ?? citation.effectiveDate}
            officialUrl={viewer?.official_url ?? citation.officialUrl}
            publication={viewer?.publication ?? undefined}
          >
            <DocumentLegalDetails
              relations={viewer?.relations}
              provisionEffects={viewer?.provision_effects}
              annotations={viewer?.source_annotations}
              officialTitleConflict={viewer?.official_title_conflict}
            />
          </LegalStatusCard>
        </div>}
      </div>
    </>
  );
}
