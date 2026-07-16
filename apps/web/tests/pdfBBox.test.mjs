import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";

import { bboxToViewportPercent } from "../src/lib/pdfBBox.ts";

const artifactPath = path.resolve(process.cwd(), "../../data/final/uud/bbox_registry.jsonl");
const rows = fs.readFileSync(artifactPath, "utf8").trim().split("\n").map(JSON.parse);

function viewport(box, scale = 1) {
  return {
    width: box.page_width * scale,
    height: box.page_height * scale,
    convertToViewportRectangle: ([x0, y0, x1, y1]) => [x0 * scale, (box.page_height - y0) * scale, x1 * scale, (box.page_height - y1) * scale],
  };
}

test("all exact corpus bboxes pass the frontend transform at zoom", () => {
  const exact = rows.filter((row) => row.bbox_precision === "exact" && row.viewer_highlightable === true);
  assert.equal(exact.length, 1566);
  for (const box of exact) {
    for (const scale of [0.75, 1, 2]) {
      const result = bboxToViewportPercent(box, viewport(box, scale));
      assert.equal(result.ok, true, box.bbox_id);
      assert.ok(Math.abs(result.left - 100 * box.x0 / box.page_width) <= 1e-9, box.bbox_id);
      assert.ok(Math.abs(result.top - 100 * box.y0 / box.page_height) <= 1e-9, box.bbox_id);
    }
  }
});

test("page-only, malformed, and unsupported rotation bboxes fail closed", () => {
  const exact = rows.find((row) => row.bbox_precision === "exact" && row.viewer_highlightable === true);
  assert.equal(bboxToViewportPercent({ ...exact, viewer_highlightable: false }, viewport(exact)).ok, false);
  assert.equal(bboxToViewportPercent({ ...exact, bbox_precision: "page_grounded_only" }, viewport(exact)).ok, false);
  assert.equal(bboxToViewportPercent({ ...exact, page_width: undefined }, viewport(exact)).ok, false);
  assert.equal(bboxToViewportPercent({ ...exact, x1: exact.x0 }, viewport(exact)).ok, false);
  assert.equal(bboxToViewportPercent({ ...exact, page_rotation: 90 }, viewport(exact)).ok, false);
});
