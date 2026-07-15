export interface PdfBBox {
  bbox_id?: string;
  page_number?: number;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  coordinate_space?: string;
  coordinate_origin?: string;
  page_width?: number;
  page_height?: number;
  page_rotation?: number;
  page_box_basis?: string;
  transform_version?: string;
  viewer_highlightable?: boolean;
  bbox_precision?: string;
}
export interface PdfViewportLike {
  width: number;
  height: number;
  convertToViewportRectangle(rect: [number, number, number, number]): number[];
}
export type BBoxViewportResult =
  | { ok: true; left: number; top: number; width: number; height: number }
  | { ok: false; reason: string };

export function bboxToViewportPercent(box: PdfBBox, viewport: PdfViewportLike): BBoxViewportResult {
  if (box.viewer_highlightable !== true || box.bbox_precision !== "exact") {
    return { ok: false, reason: "not_highlightable" };
  }
  if (
    box.coordinate_space !== "pdf_user_space" ||
    box.coordinate_origin !== "top_left" ||
    box.transform_version !== "pymupdf_top_left_v1" ||
    box.page_box_basis !== "media_box" ||
    box.page_rotation !== 0
  ) {
    return { ok: false, reason: "invalid_coordinate_metadata" };
  }
  const values = [box.x0, box.y0, box.x1, box.y1, box.page_width, box.page_height];
  if (
    !values.every((value) => typeof value === "number" && Number.isFinite(value)) ||
    !box.page_width ||
    !box.page_height ||
    box.x0! >= box.x1! ||
    box.y0! >= box.y1! ||
    box.x0! < 0 ||
    box.y0! < 0 ||
    box.x1! > box.page_width ||
    box.y1! > box.page_height ||
    viewport.width <= 0 ||
    viewport.height <= 0
  ) {
    return { ok: false, reason: "invalid_bbox" };
  }
  const rect = viewport.convertToViewportRectangle([box.x0!, box.page_height - box.y1!, box.x1!, box.page_height - box.y0!]);
  if (rect.length !== 4 || !rect.every(Number.isFinite)) return { ok: false, reason: "invalid_viewport_transform" };
  const [a, b, c, d] = rect as [number, number, number, number];
  const left = Math.min(a, c);
  const right = Math.max(a, c);
  const top = Math.min(b, d);
  const bottom = Math.max(b, d);
  return {
    ok: true,
    left: (100 * left) / viewport.width,
    top: (100 * top) / viewport.height,
    width: (100 * (right - left)) / viewport.width,
    height: (100 * (bottom - top)) / viewport.height,
  };
}
