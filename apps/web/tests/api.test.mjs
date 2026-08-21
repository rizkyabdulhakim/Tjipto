import assert from "node:assert/strict";
import { test } from "node:test";

import {
  answerTextOrFallback,
  mapAskResponseToCitations,
  mapAskResponseToDocumentSource,
  mapAskResponseToDocumentSources,
  mapAskResponseToSupportGroups,
  mapSearchResultToCitation,
} from "../src/lib/api.ts";

const legalSupport = {
  public_support_id: "support_1",
  authority_kind: "legal_citation",
  citation_final: true,
  support_kind: "legal_unit",
  fact_kind: "legal_text",
  label: "Pasal 16",
  text: "Pasal 16\nPresiden membentuk dewan.",
  source_label: "UUD 1945",
  source_status_label: "Naskah Berlaku",
  page_numbers: [9],
  citation: { number: 1, text: "Undang-Undang Dasar 1945, Pasal 16, hlm. 9", official_url: "https://example.invalid", citation_final: true },
  viewer_target: { public_target_id: "target_1", can_resolve: true, page_numbers: [9] },
};

test("maps the closed public support contract to an opaque viewer target", () => {
  const citations = mapAskResponseToCitations({ kind: "answer", status: "answer_ready", supports: [legalSupport] });
  assert.equal(citations.length, 1);
  assert.equal(citations[0].publicTargetId, "target_1");
  assert.equal(citations[0].authorityKind, "legal_citation");
  assert.equal(citations[0].citationFinal, true);
  assert.equal(citations[0].citationNumber, 1);
  assert.match(citations[0].citationText, /Pasal 16/);
  assert.equal(citations[0].factKind, "legal_text");
  assert.equal("sourceDocumentId" in citations[0], false);
});

test("nonlegal support remains clickable without becoming a legal quotation", () => {
  const support = { ...legalSupport, public_support_id: "support_2", authority_kind: "metadata_source", citation_final: false, support_kind: "metadata_source", fact_kind: "source_fact", viewer_target: { public_target_id: "target_2", can_resolve: true } };
  const citations = mapAskResponseToCitations({ kind: "answer", status: "answer_ready", supports: [support] });
  assert.equal(citations[0].authorityKind, "metadata_source");
  assert.equal(citations[0].citationFinal, false);
});

test("keeps grouped members independently targetable", () => {
  const second = { ...legalSupport, public_support_id: "support_3", viewer_target: { public_target_id: "target_3", can_resolve: true } };
  const groups = mapAskResponseToSupportGroups({
    kind: "answer", status: "answer_ready",
    support_groups: [{ public_group_id: "group_1", group_kind: "role_members", label: "Wakil Ketua", summary: "Perubahan Pertama", member_count: 2, members: [{ ...legalSupport, authority_kind: "metadata_source", citation_final: false }, { ...second, authority_kind: "metadata_source", citation_final: false }] }],
  });
  assert.equal(groups[0].members.length, 2);
  assert.deepEqual(groups[0].members.map((member) => member.publicTargetId), ["target_1", "target_3"]);
});

test("keeps grouped amendment relations typed as provenance", () => {
  const relation = {
    ...legalSupport,
    public_support_id: "relation_1",
    authority_kind: "instrument_provenance",
    citation_final: false,
    support_kind: "article_relation",
    fact_kind: "article_relation",
  };
  const groups = mapAskResponseToSupportGroups({
    kind: "answer", status: "answer_ready",
    support_groups: [{ public_group_id: "relations", group_kind: "article_relation_members", label: "Perubahan Pertama", summary: "8 ketentuan", member_count: 1, members: [relation] }],
  });
  assert.equal(groups[0].kind, "trace");
});

test("maps support categories only from typed authority", () => {
  const supports = [
    { ...legalSupport, public_support_id: "meta", authority_kind: "metadata_source", citation_final: false },
    { ...legalSupport, public_support_id: "structure", authority_kind: "structural_context", citation_final: false },
    { ...legalSupport, public_support_id: "trace", authority_kind: "source_anomaly", citation_final: false },
  ];
  const groups = mapAskResponseToSupportGroups({ kind: "answer", status: "answer_ready", supports });
  assert.deepEqual(groups.map((group) => group.kind), ["metadata", "structure", "trace"]);
  assert.deepEqual(groups.map((group) => group.title), ["Sumber Dokumen", "Struktur Dokumen", "Catatan Sumber"]);
});

test("maps search and document source through public targets only", () => {
  const search = mapSearchResultToCitation({
    title: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
    legal_identity: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
    issuer: "MPR RI",
    legal_status: "Berlaku",
    document_role: "Naskah Konsolidasi",
    official_url: "https://example.invalid",
    viewer_target: { public_target_id: "search_1" },
  }, 0);
  assert.equal(search?.publicTargetId, "search_1");
  assert.equal(search?.viewerMode, "catalog");
  assert.equal(search?.documentTitle, "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945");
  assert.doesNotMatch(search?.documentTitle, /Nomor 1945 Tahun 2002/);
  const document = mapAskResponseToDocumentSource({ kind: "document", status: "answer_ready", document: { label: "Dokumen", viewer_target: { public_target_id: "document_1" } } });
  assert.equal(document?.publicTargetId, "document_1");
  assert.equal(document?.viewerMode, "document");
});

test("maps a verified document collection to independently openable cards", () => {
  const documents = mapAskResponseToDocumentSources({
    kind: "documents",
    status: "answer_ready",
    documents: [
      { title: "Naskah Asli", document_role: "Naskah Asli", viewer_target: { public_target_id: "original" } },
      { title: "Perubahan Pertama", document_role: "Amandemen", viewer_target: { public_target_id: "amendment-1" } },
    ],
  });
  assert.deepEqual(documents.map((document) => document.publicTargetId), ["original", "amendment-1"]);
  assert.ok(documents.every((document) => document.viewerMode === "document"));
});

test("uses a safe answer fallback only when the public answer is empty", () => {
  assert.equal(answerTextOrFallback({ kind: "answer", status: "answer_ready", answer: "Jawaban" }), "Jawaban");
  assert.match(answerTextOrFallback({ kind: "unavailable", status: "insufficient_evidence" }), /Bukti tidak cukup/);
});
