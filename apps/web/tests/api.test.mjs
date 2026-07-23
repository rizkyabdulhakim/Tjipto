import assert from "node:assert/strict";
import { test } from "node:test";

import {
  answerTextOrFallback,
  mapAskResponseToCitations,
  mapAskResponseToDocumentSource,
  mapAskResponseToSupportGroups,
  mapAskResponseToSupportItems,
  mapSearchResultToCitation,
} from "../src/lib/api.ts";

const legalSupport = {
  public_support_id: "support_1",
  support_kind: "legal_unit",
  panel_section: "Kutipan Relevan",
  fact_kind: "legal_text",
  label: "Pasal 16",
  text: "Pasal 16\nPresiden membentuk dewan.",
  layout_lines: [{ text: "Pasal 16", line_order: 0, paragraph_id: "0", alignment: "center", indent: 0 }],
  copy_text: "Pasal 16\n\nPresiden membentuk dewan.",
  source_label: "UUD 1945",
  source_role: "current_consolidated",
  page_numbers: [9],
  legal_citation_available: true,
  relevant_quote_eligible: true,
  viewer_target: { public_target_id: "target_1", can_resolve: true, page_numbers: [9] },
};

test("maps the closed public support contract to an opaque viewer target", () => {
  const citations = mapAskResponseToCitations({ status: "answer_ready", supports: [legalSupport] });
  assert.equal(citations.length, 1);
  assert.equal(citations[0].publicTargetId, "target_1");
  assert.equal(citations[0].relevantQuoteEligible, true);
  assert.equal(citations[0].copyText, legalSupport.copy_text);
  assert.equal("sourceDocumentId" in citations[0], false);
});

test("nonlegal support remains clickable without becoming a legal quotation", () => {
  const support = { ...legalSupport, public_support_id: "support_2", support_kind: "metadata_source", panel_section: "Sumber Dokumen", legal_citation_available: false, relevant_quote_eligible: false, viewer_target: { public_target_id: "target_2", can_resolve: true } };
  const citations = mapAskResponseToCitations({ status: "answer_ready", supports: [support] });
  const items = mapAskResponseToSupportItems({ status: "answer_ready", supports: [support] });
  assert.equal(citations[0].panelSection, "Sumber Dokumen");
  assert.equal(citations[0].relevantQuoteEligible, false);
  assert.equal(items.metadata[0].publicTargetId, "target_2");
});

test("keeps grouped members independently targetable", () => {
  const second = { ...legalSupport, public_support_id: "support_3", viewer_target: { public_target_id: "target_3", can_resolve: true } };
  const groups = mapAskResponseToSupportGroups({
    status: "answer_ready",
    support_groups: [{ public_group_id: "group_1", panel_section: "Sumber Dokumen", label: "Wakil Ketua", summary: "Perubahan Pertama", member_count: 2, members: [{ ...legalSupport, panel_section: "Sumber Dokumen", legal_citation_available: false, relevant_quote_eligible: false }, { ...second, panel_section: "Sumber Dokumen", legal_citation_available: false, relevant_quote_eligible: false }] }],
  });
  assert.equal(groups[0].members.length, 2);
  assert.deepEqual(groups[0].members.map((member) => member.publicTargetId), ["target_1", "target_3"]);
});

test("maps search and document source through public targets only", () => {
  const search = mapSearchResultToCitation({ title: "UUD 1945", snippet: "Dokumen sumber", page_numbers: [1], viewer_target: { public_target_id: "search_1" } }, 0);
  assert.equal(search?.publicTargetId, "search_1");
  const document = mapAskResponseToDocumentSource({ status: "answer_ready", document_source: { label: "Dokumen", viewer_target: { public_target_id: "document_1" } } });
  assert.equal(document?.publicTargetId, "document_1");
});

test("uses a safe answer fallback only when the public answer is empty", () => {
  assert.equal(answerTextOrFallback({ status: "answer_ready", answer: "Jawaban" }), "Jawaban");
  assert.match(answerTextOrFallback({ status: "insufficient_evidence" }), /Bukti tidak cukup/);
});
