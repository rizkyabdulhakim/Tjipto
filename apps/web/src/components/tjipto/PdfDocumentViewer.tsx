import type { RefObject } from "react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import * as pdfjs from "pdfjs-dist";
import type { PDFDocumentProxy, PDFPageProxy } from "pdfjs-dist/types/src/display/api";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
import { FileText } from "lucide-react";
import type { ViewerPayload } from "../../lib/api";
import { pdfAccessUrl } from "../../lib/api";
import { bboxToViewportPercent } from "../../lib/pdfBBox";
import { canvasBackingStore, clamp, fitWidthScale, isRenderCancellation, RenderTaskOwner, visiblePageWindow } from "../../lib/pdfViewer";

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

export function RenderedViewer({
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
    const update = () => setAvailableWidth(Math.max(1, Math.min(960, node.clientWidth - 24)));
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
      className="w-max min-w-full bg-[#121212] px-3 py-4 space-y-4"
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

export function UnavailableViewer({
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
  const viewerReady = viewer?.status === "viewer_payload_ready";
  return (
    <div
      className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 py-10 text-center"
    >
      <FileText size={28} className="text-[var(--tj-text-muted)]" />
      <div style={{ fontSize: 14, fontWeight: 700, color: "var(--tj-text-primary)" }}>
        {loading ? "Memuat naskah resmi" : "Naskah belum dapat ditampilkan"}
      </div>
      <div style={{ fontSize: 13, color: "var(--tj-text-secondary)", lineHeight: "20px" }}>
        {viewerReady
          ? `${location} ditemukan pada halaman ${pageNumber}, tetapi tampilan halaman belum tersedia.`
          : "Sumber resmi belum dapat ditampilkan pada panel ini."}
      </div>
    </div>
  );
}
