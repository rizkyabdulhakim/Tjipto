import assert from "node:assert/strict";
import { test } from "node:test";

import {
  canvasBackingStore,
  fitWidthScale,
  isRenderCancellation,
  RenderTaskOwner,
  visiblePageWindow,
} from "../src/lib/pdfViewer.ts";

test("fits logical PDF viewport and scales only the backing store for DPR", () => {
  assert.equal(fitWidthScale(500, 400, 1), 0.8);
  assert.equal(fitWidthScale(500, 400, 2), 1.6);
  assert.deepEqual(canvasBackingStore(400, 600, 2), { width: 800, height: 1200, ratio: 2 });
});

test("keeps canvases to visible pages and one adjacent page", () => {
  assert.deepEqual([...visiblePageWindow([4], 8)], [3, 4, 5]);
  assert.deepEqual([...visiblePageWindow([1, 2], 8)], [1, 2, 3]);
  assert.deepEqual([...visiblePageWindow([8], 8)], [7, 8]);
});

test("recognizes PDF.js cancellation without hiding other failures", () => {
  const cancelled = new Error("cancelled");
  cancelled.name = "RenderingCancelledException";
  assert.equal(isRenderCancellation(cancelled), true);
  assert.equal(isRenderCancellation(new Error("network")), false);
});

test("owns one render task and cancels replacement and teardown", () => {
  const owner = new RenderTaskOwner();
  const first = { cancelled: false, cancel() { this.cancelled = true; } };
  const second = { cancelled: false, cancel() { this.cancelled = true; } };
  owner.replace(first);
  owner.replace(second);
  assert.equal(first.cancelled, true);
  assert.equal(owner.isCurrent(second), true);
  owner.cancel();
  assert.equal(second.cancelled, true);
  assert.equal(owner.isCurrent(second), false);
});
