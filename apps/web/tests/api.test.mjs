import assert from "node:assert/strict";
import { test } from "node:test";

import {
  answerTextOrFallback,
  mapAskResponseToCitations,
  mapAskResponseToDocumentSource,
  mapAskResponseToSupportItems,
  mapSearchResultToCitation,
} from "../src/lib/api.ts";

test("document search result opens document viewer mode", () => {
  const citation = mapSearchResultToCitation({
    corpus_id: "uud",
    source_document_id: "uud::current_consolidated",
    document_id: "uud::current_consolidated",
    document_title: "UUD 1945",
    source_url: "https://peraturan.bpk.go.id/Details/101646/uud-no--",
    page_numbers: [1],
    snippet: "Dokumen sumber terverifikasi",
    status: "document",
  }, 0);

  assert.equal(citation?.viewerMode, "document");
  assert.equal(citation?.documentId, "uud::current_consolidated");
  assert.equal(citation?.sourceDocumentId, "uud::current_consolidated");
  assert.equal(citation?.pageNumber, 1);
  assert.equal(citation?.sourceUrl, "https://peraturan.bpk.go.id/Details/101646/uud-no--");
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

test("scoped source answer opens the full document without citation geometry", () => {
  const citation = mapAskResponseToDocumentSource({
    status: "answer_ready",
    answer_type: "source_document",
    document_source: {
      source_document_id: "uud::amendment_1_historical",
      source_role: "amendment_1_historical",
      temporal_context: "amendment_1_historical",
      document_title: "Perubahan Pertama UUD 1945",
      viewer_target: { action: "open_document", source_document_id: "uud::amendment_1_historical" },
    },
    citations: [],
    viewer_refs: [],
  });

  assert.equal(citation?.viewerMode, "document");
  assert.equal(citation?.sourceDocumentId, "uud::amendment_1_historical");
  assert.equal(citation?.excerpt, "");
  assert.deepEqual(mapAskResponseToCitations({ status: "answer_ready", answer_type: "source_document" }), []);
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

test("relation citation preserves viewer source identity and proof layers", () => {
  const citations = mapAskResponseToCitations({
    status: "answer_ready",
    citations: [{
      evidence_id: "evidence_rename",
      quoted_text: "Pasal 25E menjadi Pasal 25A",
      support_kind: "legal_unit",
      relevant_quote_eligible: true,
      authority_kind: "legal_citation",
      citation_final: false,
      page_numbers: [1],
    }],
    viewer_refs: [{
      evidence_id: "evidence_rename",
      source_document_id: "uud::amendment_4_historical",
      page_numbers: [1],
      source_proof_text_span_ids: ["span_old", "span_transition", "span_new"],
      source_proof_bbox_refs: ["bbox_old", "bbox_transition", "bbox_new"],
      target_text_span_ids: ["span_new"],
      target_bbox_refs: ["bbox_new"],
      can_resolve: true,
    }],
  });

  assert.equal(citations[0].sourceDocumentId, "uud::amendment_4_historical");
  assert.deepEqual(citations[0].relationSourceProofBBoxRefs, ["bbox_old", "bbox_transition", "bbox_new"]);
  assert.deepEqual(citations[0].relationTargetBBoxRefs, ["bbox_new"]);
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

test("article relation support preserves source proof and target precision wording", () => {
  const support = mapAskResponseToSupportItems({
    status: "answer_ready",
    article_amendment_relations: [{
      relation_id: "rename_25e_25a",
      relation_type: "RENAMES",
      source_reference: "Pasal 25E",
      target_reference: "Pasal 25A",
      evidence_id: "evidence_rename",
      source_proof_text_span_ids: ["span_old", "span_transition", "span_new"],
      source_proof_bbox_refs: ["bbox_old", "bbox_transition", "bbox_new"],
      target_precision: "target_local",
      support_class: "exact_article_relation",
      viewer_highlightable: true,
    }],
  });

  assert.equal(support.articleRelations.length, 1);
  assert.match(support.articleRelations[0].label, /Pasal 25E.*Pasal 25A/);
  assert.match(support.articleRelations[0].detail, /Source proof exact/);
  assert.match(support.articleRelations[0].detail, /target target_local/);
});
