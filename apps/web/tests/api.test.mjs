import assert from "node:assert/strict";
import { test } from "node:test";

import { mapAskResponseToCitations, mapSearchResultToCitation } from "../src/lib/api.ts";

test("document search result opens document viewer mode", () => {
  const citation = mapSearchResultToCitation({
    corpus_id: "uud",
    source_document_id: "uud::current_consolidated",
    document_id: "uud::current_consolidated",
    document_title: "UUD 1945",
    page_numbers: [1],
    snippet: "Dokumen sumber terverifikasi",
    status: "document",
  }, 0);

  assert.equal(citation?.viewerMode, "document");
  assert.equal(citation?.documentId, "uud::current_consolidated");
  assert.equal(citation?.sourceDocumentId, "uud::current_consolidated");
  assert.equal(citation?.pageNumber, 1);
});

test("evidence search result keeps evidence viewer mode", () => {
  const citation = mapSearchResultToCitation({
    corpus_id: "uud",
    evidence_id: "evidence_1",
    source_document_id: "uud::current_consolidated",
    snippet: "Pasal 1",
    status: "evidence",
    viewer_ref: { page_numbers: [3] },
  }, 0);

  assert.equal(citation?.viewerMode, "evidence");
  assert.equal(citation?.documentId, "evidence_1");
  assert.equal(citation?.pageNumber, 3);
});

test("metadata and trace support are not mapped as exact citations", () => {
  const citations = mapAskResponseToCitations({
    status: "answer_ready",
    answer: "19 Oktober 1999",
    metadata_support: [{ evidence_id: "meta_1", field: "enactment_date", answer: "19 Oktober 1999" }],
    trace_support: [{ relation_id: "trace_1", target_citation: "Pasal 31", citation_available: false }],
    document_relations: [{ relation_id: "doc_1", relation_type: "AMENDED_BY", highlightable: false }],
  });

  assert.deepEqual(citations, []);
});
