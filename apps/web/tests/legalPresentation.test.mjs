import assert from "node:assert/strict";
import { test } from "node:test";

import {
  documentRole,
  legalIdentity,
  legalStatus,
  numberAndYear,
} from "../src/lib/legalPresentation.ts";

test("formats constitutional and regulation identities without inventing a UUD number", () => {
  const constitution = {
    official_title: "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
    document_type: "Undang-Undang Dasar",
  };
  assert.equal(legalIdentity(constitution), constitution.official_title);
  assert.equal(numberAndYear(constitution), undefined);
  assert.equal(legalIdentity({ document_type: "Peraturan Presiden", number: "98", year: "2020" }), "Peraturan Presiden Nomor 98 Tahun 2020");
});

test("fails closed for unknown status and role values", () => {
  assert.equal(legalStatus("unknown_internal_status"), "Belum Diverifikasi");
  assert.equal(documentRole("unknown_internal_role"), undefined);
  assert.equal(documentRole("Naskah Konsolidasi"), "Naskah Konsolidasi");
});
