import assert from "node:assert/strict";
import { test } from "node:test";

import { answerTextOrFallback, mapAskResponseToCitations, mapAskResponseToSupportItems, mapSearchResultToCitation } from "../src/lib/api.ts";

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

test("limited answer keeps backend answer text instead of fallback", () => {
  assert.equal(
    answerTextOrFallback({
      status: "limited_answer",
      answer: "Catatan konflik sumber tersedia.",
    }),
    "Catatan konflik sumber tersedia.",
  );
});

test("insufficient evidence without answer still falls back safely", () => {
  assert.match(
    answerTextOrFallback({
      status: "insufficient_evidence",
      answer: "",
    }),
    /Bukti tidak cukup/,
  );
});

test("source-conflict exact provenance citations are labeled as audit provenance", () => {
  const citations = mapAskResponseToCitations({
    status: "limited_answer",
    route: "source_anomaly_explanation",
    answer_scope: "source_conflict_exact_provenance",
    warnings: ["source_conflict_not_final_legal_authority"],
    citations: [
      {
        evidence_id: "evidence_1",
        source_document_id: "uud::amendment_2_historical",
        quoted_text: "Pasal 25E",
        label: "BAB IXA / Pasal 25E",
        page_numbers: [3],
      },
    ],
    viewer_refs: [
      {
        evidence_id: "evidence_1",
        page_numbers: [3],
        can_resolve: true,
      },
    ],
  });

  assert.equal(citations.length, 1);
  assert.equal(citations[0].authorityKind, "source_conflict_provenance");
  assert.equal(citations[0].authorityLabel, "Jejak audit sumber");
});

test("non-resolvable provenance citations are not mapped as clickable citations", () => {
  const citations = mapAskResponseToCitations({
    status: "limited_answer",
    route: "source_anomaly_explanation",
    answer_scope: "source_conflict_exact_provenance",
    warnings: ["source_conflict_not_final_legal_authority"],
    citations: [
      {
        evidence_id: "evidence_1",
        source_document_id: "uud::amendment_2_historical",
        quoted_text: "Pasal 25E",
      },
    ],
    viewer_refs: [
      {
        evidence_id: "evidence_1",
        can_resolve: false,
      },
    ],
  });

  assert.deepEqual(citations, []);
});

test("trace-only source conflict stays support-only and non-clickable", () => {
  const support = mapAskResponseToSupportItems({
    status: "limited_answer",
    route: "source_anomaly_explanation",
    answer_scope: "source_conflict_trace",
    trace_support: [
      {
        source_conflict_id: "conflict_1",
        support_class: "source_conflict_trace",
        classification: "source_pdf_contains_pasal_iii_conflict",
      },
    ],
  });

  assert.equal(support.trace.length, 1);
  assert.equal(support.trace[0].clickable, false);
  assert.match(String(support.trace[0].detail), /tidak dapat di-highlight/i);
});
