import assert from "node:assert/strict";
import { test } from "node:test";

import {
  askLegal,
  answerTextOrFallback,
  mapAskResponseToCitations,
  mapAskResponseToDocumentSource,
  mapAskResponseToSupportItems,
  mapSearchResultToCitation,
} from "../src/lib/api.ts";

test("document search result opens its opaque viewer target", () => {
  const citation = mapSearchResultToCitation({
    corpus_id: "uud",
    document_title: "UUD 1945",
    page_numbers: [1],
    snippet: "Dokumen sumber terverifikasi",
    status: "document",
    viewer_target: { target: "search_target_1" },
  }, 0);

  assert.equal(citation?.viewerMode, "document");
  assert.equal(citation?.documentId, "search_target_1");
  assert.equal(citation?.sourceDocumentId, undefined);
  assert.equal(citation?.pageNumber, 1);
  assert.equal(citation?.sourceUrl, "");
});

test("evidence search result keeps its opaque viewer target", () => {
  const citation = mapSearchResultToCitation({
    corpus_id: "uud",
    snippet: "Pasal 1",
    status: "evidence",
    viewer_target: { target: "search_target_2", page_numbers: [3] },
  }, 0);

  assert.equal(citation?.viewerMode, "evidence");
  assert.equal(citation?.documentId, "search_target_2");
  assert.equal(citation?.pageNumber, 3);
});

test("scoped source answer uses an opaque viewer target", () => {
  const citation = mapAskResponseToDocumentSource({
    status: "answer_ready",
    answer_type: "source_document",
    document_source: {
      source_role: "amendment_1_historical",
      temporal_context: "amendment_1_historical",
      document_title: "Perubahan Pertama UUD 1945",
      viewer_target: { action: "open_document", target: "viewer_target_1" },
    },
    citations: [],
    viewer_refs: [],
  });

  assert.equal(citation?.viewerMode, "document");
  assert.equal(citation?.documentId, "viewer_target_1");
  assert.equal(citation?.excerpt, "");
  assert.deepEqual(mapAskResponseToCitations({ status: "answer_ready", answer_type: "source_document" }), []);
});

test("a response without public supports has no citations", () => {
  const citations = mapAskResponseToCitations({
    status: "answer_ready",
    answer: "19 Oktober 1999",
    document_relations: [{ relation_id: "doc_1", relation_type: "AMENDED_BY", highlightable: false }],
  });

  assert.deepEqual(citations, []);
});

test("metadata provenance is not mapped as a relevant legal quotation", () => {
  const citations = mapAskResponseToCitations({
    status: "answer_ready",
    route: "metadata_fact",
    citations: [
      {
        evidence_id: "meta_1",
        source_document_id: "uud::amendment_1_historical",
        source_url: "https://peraturan.bpk.go.id/Details/101646/uud-no--",
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

  assert.deepEqual(citations, []);
});

test("relation citation preserves opaque viewer source identity", () => {
  const citations = mapAskResponseToCitations({
    status: "answer_ready",
    supports: [{
      support_id: "evidence_rename",
      support_kind: "legal_unit",
      panel_section: "Kutipan Relevan",
      display_label: "Pasal 25E",
      display_text: "Pasal 25E menjadi Pasal 25A",
      copy_text: "Pasal 25E menjadi Pasal 25A",
      source_document: "uud::amendment_4_historical",
      source_role: "amendment_4_historical",
      page_numbers: [1],
      legal_citation_available: true,
      linkable: true,
      highlightable: true,
      viewer_target: {
        source_document_id: "uud::amendment_4_historical",
        page_numbers: [1],
        can_resolve: true,
      },
    }],
  });

  assert.equal(citations[0].sourceDocumentId, "uud::amendment_4_historical");
  assert.equal(citations[0].viewerRefId, "evidence_rename");
  assert.equal(citations[0].viewerTarget?.can_resolve, true);
});

test("exact non-legal source text stays clickable outside relevant quotations", () => {
  const citations = mapAskResponseToCitations({
    status: "answer_ready",
    supports: [{
      support_id: "source_support_hash",
      support_kind: "source_text",
      panel_section: "Sumber Dokumen",
      display_label: "PERUBAHAN PERTAMA",
      display_text: "PERUBAHAN PERTAMA",
      copy_text: "PERUBAHAN PERTAMA",
      source_document: "uud::amendment_1_historical",
      source_role: "amendment_1_historical",
      page_numbers: [1],
      legal_citation_available: false,
      relevant_quote_eligible: false,
      linkable: true,
      highlightable: true,
      viewer_target: { can_resolve: true, source_document_id: "uud::amendment_1_historical", page_numbers: [1] },
    }],
  });

  assert.equal(citations.length, 1);
  assert.equal(citations[0].panelSection, "Sumber Dokumen");
  assert.equal(citations[0].relevantQuoteEligible, false);
  assert.equal(citations[0].viewerTarget?.can_resolve, true);
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

test("source-conflict provenance is not mapped as a relevant legal quotation", () => {
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

  assert.deepEqual(citations, []);
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

test("source anomaly provenance stays outside relevant quotations", () => {
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

  assert.deepEqual(citations, []);
});

test("article relation support uses the source-note panel", () => {
  const support = mapAskResponseToSupportItems({
    status: "answer_ready",
    supports: [{
      support_id: "evidence_rename",
      support_kind: "article_relation",
      panel_section: "Catatan Sumber",
      display_label: "RENAMES",
      display_text: "Pasal 25E menjadi Pasal 25A",
      linkable: true,
      highlightable: true,
      viewer_target: { can_resolve: true },
    }],
  });

  assert.equal(support.trace.length, 1);
  assert.equal(support.trace[0].label, "RENAMES");
  assert.equal(support.trace[0].clickable, true);
});

test("ask sends the original question with its selected source role", async () => {
  const originalFetch = globalThis.fetch;
  let payload;
  globalThis.fetch = async (_url, options) => {
    payload = JSON.parse(options.body);
    return new Response(JSON.stringify({ status: "answer_ready" }), { status: 200 });
  };
  try {
    await askLegal("ketua MPR", { source_role: "amendment_1_historical" });
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.deepEqual(payload, {
    query: "ketua MPR",
    filters: { source_role: "amendment_1_historical" },
  });
});
