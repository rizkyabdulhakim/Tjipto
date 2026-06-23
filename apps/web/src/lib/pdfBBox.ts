export interface PdfBBox {
  bbox_id?: string;
  page_number?: number;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
}

export interface PdfViewportLike {
  width: number;
  height: number;
  convertToViewportRectangle(rect: [number, number, number, number]): number[];
}

export function bboxToViewportPercent(box: PdfBBox, viewport: PdfViewportLike) {
  const rect = viewport.convertToViewportRectangle([
    box.x0 ?? 0,
    box.y0 ?? 0,
    box.x1 ?? 0,
    box.y1 ?? 0,
  ]);
  const left = Math.min(rect[0] ?? 0, rect[2] ?? 0);
  const right = Math.max(rect[0] ?? 0, rect[2] ?? 0);
  const top = Math.min(rect[1] ?? 0, rect[3] ?? 0);
  const bottom = Math.max(rect[1] ?? 0, rect[3] ?? 0);
  return {
    left: percent(left, viewport.width),
    top: percent(top, viewport.height),
    width: percent(right - left, viewport.width),
    height: percent(bottom - top, viewport.height),
  };
}

function percent(value: number, total: number) {
  return total > 0 ? (100 * value) / total : 0;
}
