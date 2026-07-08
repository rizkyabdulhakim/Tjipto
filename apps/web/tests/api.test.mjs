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

test("exact metadata provenance is mapped as non-final metadata source", () => {
  const citations = mapAskResponseToCitations({
    status: "answer_ready",
    route: "metadata_fact",
    citations: [
      {
        evidence_id: "meta_1",
        source_document_id: "uud::amendment_1_historical",
        quoted_text: "Pada tanggal 19 Oktober 1999",
        label: "Metadata amendment_1_historical: date",
        authority_kind: "metadata_source",
        authority_label: "Metadata sumber",
        citation_final: false,
        page_numbers: [3],
      },
    ],
    viewer_refs: [
      {
        evidence_id: "meta_1",
        page_numbers: [3],
        can_resolve: true,
      },
    ],
  });

  assert.equal(citations.length, 1);
  assert.equal(citations[0].authorityKind, "metadata_source");
  assert.equal(citations[0].authorityLabel, "Metadata sumber");
  assert.equal(citations[0].citationFinal, false);
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
    citations: [
      {
        evidence_id: "evidence_1",
        source_document_id: "uud::amendment_2_historical",
        quoted_text: "Pasal 25E",
        label: "BAB IXA / Pasal 25E",
        page_numbers: [3],
        authority_kind: "source_conflict_provenance",
        authority_label: "Jejak audit sumber",
        citation_final: false,
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

test("exact source anomaly provenance stays non-final but clickable when viewer refs resolve", () => {
  const citations = mapAskResponseToCitations({
    status: "limited_answer",
    route: "source_anomaly_explanation",
    citations: [
      {
        evidence_id: "conflict_1",
        source_document_id: "uud::amendment_4_historical",
        quoted_text: "Pasal III",
        page_numbers: [5],
        authority_kind: "source_anomaly",
        authority_label: "Source anomaly",
        citation_final: false,
      },
    ],
    viewer_refs: [
      {
        evidence_id: "conflict_1",
        page_numbers: [5],
        can_resolve: true,
      },
    ],
  });

  assert.equal(citations.length, 1);
  assert.equal(citations[0].authorityKind, "source_anomaly");
  assert.equal(citations[0].authorityLabel, "Source anomaly");
  assert.equal(citations[0].citationFinal, false);
});
