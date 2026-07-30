import assert from "node:assert/strict";
import { test } from "node:test";

import {
  answerTextOrFallback,
  mapAskResponseToCitations,
  mapAskResponseToDocumentSource,
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
  source_role: "current_consolidated",
  page_numbers: [9],
  viewer_target: { public_target_id: "target_1", can_resolve: true, page_numbers: [9] },
};

test("maps the closed public support contract to an opaque viewer target", () => {
  const citations = mapAskResponseToCitations({ kind: "answer", status: "answer_ready", supports: [legalSupport] });
  assert.equal(citations.length, 1);
  assert.equal(citations[0].publicTargetId, "target_1");
  assert.equal(citations[0].authorityKind, "legal_citation");
  assert.equal(citations[0].citationFinal, true);
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
  const search = mapSearchResultToCitation({ title: "UUD 1945", snippet: "Dokumen sumber", page_numbers: [1], viewer_target: { public_target_id: "search_1" } }, 0);
  assert.equal(search?.publicTargetId, "search_1");
  const document = mapAskResponseToDocumentSource({ kind: "document", status: "answer_ready", document: { label: "Dokumen", viewer_target: { public_target_id: "document_1" } } });
  assert.equal(document?.publicTargetId, "document_1");
});

test("uses a safe answer fallback only when the public answer is empty", () => {
  assert.equal(answerTextOrFallback({ kind: "answer", status: "answer_ready", answer: "Jawaban" }), "Jawaban");
  assert.match(answerTextOrFallback({ kind: "unavailable", status: "insufficient_evidence" }), /Bukti tidak cukup/);
});
