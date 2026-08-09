import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";

import { bboxToViewportPercent } from "../src/lib/pdfBBox.ts";
import { canvasBackingStore } from "../src/lib/pdfViewer.ts";

const artifactPath = path.resolve(process.cwd(), "../../data/final/uud/bbox_registry.jsonl");
const rows = fs.readFileSync(artifactPath, "utf8").trim().split("\n").map(JSON.parse);
const finalDir = path.dirname(artifactPath);
const jsonl = (name) => fs.readFileSync(path.join(finalDir, name), "utf8").trim().split("\n").map(JSON.parse);

function viewport(box, scale = 1) {
  return {
    width: box.page_width * scale,
    height: box.page_height * scale,
    convertToViewportPoint: (x, y) => [x * scale, (box.page_height - y) * scale],
  };
}

test("all exact corpus bboxes pass the frontend transform at zoom", () => {
  const exact = rows.filter((row) => row.bbox_precision === "exact" && row.viewer_highlightable === true);
  assert.ok(exact.length >= 1616);
  for (const box of exact) {
    for (const scale of [0.75, 1, 2]) {
      const result = bboxToViewportPercent(box, viewport(box, scale));
      assert.equal(result.ok, true, box.bbox_id);
      assert.ok(Math.abs(result.left - 100 * box.x0 / box.page_width) <= 1e-9, box.bbox_id);
      assert.ok(Math.abs(result.top - 100 * box.y0 / box.page_height) <= 1e-9, box.bbox_id);
      for (const dpr of [1, 2]) {
        const backing = canvasBackingStore(box.page_width * scale, box.page_height * scale, dpr);
        assert.equal(backing.width, Math.ceil(box.page_width * scale * dpr), box.bbox_id);
        assert.equal(backing.height, Math.ceil(box.page_height * scale * dpr), box.bbox_id);
      }
    }
  }
});

test("page-only and malformed public rectangles fail closed", () => {
  const exact = rows.find((row) => row.bbox_precision === "exact" && row.viewer_highlightable === true);
  assert.equal(bboxToViewportPercent({ ...exact, viewer_highlightable: false }, viewport(exact)).ok, false);
  assert.equal(bboxToViewportPercent({ ...exact, bbox_precision: "page_grounded_only" }, viewport(exact)).ok, false);
  assert.equal(bboxToViewportPercent({ ...exact, page_width: undefined }, viewport(exact)).ok, false);
  assert.equal(bboxToViewportPercent({ ...exact, x1: exact.x0 }, viewport(exact)).ok, false);
});

test("Pasal 7C viewer overlay highlights legal text without covering the visible source marker", () => {
  const words = jsonl("word_bboxes.jsonl");
  const proposition = jsonl("propositions.jsonl").find(
    (row) =>
      row.source_document_id === "uud::current_consolidated" &&
      row.page_numbers.length === 1 &&
      row.page_numbers[0] === 7 &&
      row.exact_quote.startsWith("Pasal 7C\n"),
  );
  assert.ok(proposition);
  const characters = words.flatMap((word) => word.characters.map((character) => ({ ...word, ...character })));
  const charactersById = new Map(characters.map((character) => [character.character_bbox_id, character]));
  const selectedPeriod = proposition.bbox_refs.map((id) => charactersById.get(id)).find((character) => character?.text === ".");
  assert.ok(selectedPeriod);
  const marker = characters.find(
    (character) =>
      character.text === "*" &&
      character.source_document_id === selectedPeriod.source_document_id &&
      character.page_number === selectedPeriod.page_number &&
      Math.min(character.x1, selectedPeriod.x1) > Math.max(character.x0, selectedPeriod.x0) &&
      Math.min(character.y1, selectedPeriod.y1) > Math.max(character.y0, selectedPeriod.y0),
  );
  assert.ok(marker, "the source marker must remain present in the extracted PDF geometry");
  const overlays = proposition.viewer_overlay.clipped_rectangle_indexes
    .map((index) => ({
      ...proposition.viewer_overlay.geometry_spaces[
        proposition.viewer_overlay.rectangles[index].geometry_space_index
      ],
      ...proposition.viewer_overlay.rectangles[index],
      character_bbox_ids: proposition.bbox_refs.slice(
        proposition.viewer_overlay.rectangles[index].selected_character_start,
        proposition.viewer_overlay.rectangles[index].selected_character_end,
      ),
      bbox_precision: "exact",
      viewer_highlightable: true,
    }))
    .filter((rectangle) => rectangle.character_bbox_ids.includes(selectedPeriod.character_bbox_id));
  assert.ok(overlays.length);
  for (const scale of [0.75, 1, 2]) {
    const markerRect = bboxToViewportPercent(
      { ...marker, bbox_precision: "exact", viewer_highlightable: true },
      viewport(marker, scale),
    );
    assert.equal(markerRect.ok, true);
    for (const overlay of overlays) {
      const overlayRect = bboxToViewportPercent(overlay, viewport(overlay, scale));
      assert.equal(overlayRect.ok, true);
      const horizontal = Math.min(overlayRect.left + overlayRect.width, markerRect.left + markerRect.width) -
        Math.max(overlayRect.left, markerRect.left);
      const vertical = Math.min(overlayRect.top + overlayRect.height, markerRect.top + markerRect.height) -
        Math.max(overlayRect.top, markerRect.top);
      assert.ok(horizontal <= 1e-9 || vertical <= 1e-9, `marker overlap at zoom ${scale}`);
    }
  }
});
