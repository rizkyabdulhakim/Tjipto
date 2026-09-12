from __future__ import annotations

from datetime import datetime, timezone
import unittest

from tjipto.contracts.legal_information import (
    ConflictKind,
    ConflictResolution,
    DocumentRelation,
    FieldState,
    LegalDocumentIdentity,
    OfficialValue,
    ProvisionEffect,
    RelationKind,
    ResolutionState,
    SourceKind,
    SourceProvenance,
    StatusAssertion,
    VerifiedValue,
)


NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)
CATALOG = SourceProvenance(SourceKind.OFFICIAL_CATALOG_PAGE, "https://example.invalid/catalog", NOW)
PDF = SourceProvenance(
    SourceKind.OFFICIAL_PDF,
    "https://example.invalid/document.pdf",
    NOW,
    "a" * 64,
    2,
    "Mengubah Pasal 1.",
)


class LegalInformationContractTest(unittest.TestCase):
    def test_source_designation_can_identify_a_document_when_issuer_is_absent_from_source(self) -> None:
        def value(text: str) -> VerifiedValue:
            return VerifiedValue(text, text.casefold(), text, FieldState.VERIFIED, PDF)

        missing = VerifiedValue(None, None, None, FieldState.NOT_FOUND_IN_SOURCE)
        identity = LegalDocumentIdentity(
            value("Undang-Undang Dasar"),
            VerifiedValue(None, None, None, FieldState.NOT_APPLICABLE),
            value("1945"),
            value("Undang-Undang Dasar Negara Republik Indonesia Tahun 1945"),
            missing,
            value("Naskah Asli"),
        )
        self.assertTrue(identity.stable_id.startswith("legal-document-"))
        with self.assertRaisesRegex(ValueError, "identity_not_verified"):
            LegalDocumentIdentity(
                identity.document_type,
                identity.number,
                identity.year,
                identity.official_title,
                missing,
            ).stable_id

    def test_verified_values_preserve_source_normalized_display_and_provenance(self) -> None:
        value = VerifiedValue("Berlaku ", "applicable", "Berlaku", FieldState.VERIFIED, CATALOG)
        self.assertEqual((value.source_value, value.normalized_value, value.display_value), ("Berlaku ", "applicable", "Berlaku"))
        with self.assertRaisesRegex(ValueError, "verified_value_requires_source_and_provenance"):
            VerifiedValue("Berlaku", "applicable", "Berlaku", FieldState.VERIFIED)

    def test_conflicts_cannot_be_silently_collapsed(self) -> None:
        values = (
            OfficialValue(
                "Judul A",
                "judul a",
                "Judul A",
                SourceProvenance(SourceKind.OFFICIAL_CATALOG_PAGE, CATALOG.reference, NOW, source_authority="BPK"),
            ),
            OfficialValue(
                "Judul B",
                "judul b",
                "Judul B",
                SourceProvenance(SourceKind.OFFICIAL_JDIH_PAGE, "https://example.invalid/jdih", NOW, source_authority="JDIH"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicting_value_collapsed"):
            VerifiedValue(
                "Judul A",
                "judul a",
                "Judul A",
                FieldState.CONFLICTING_SOURCES,
                values[0].provenance,
                values,
                ConflictKind.SOURCE_VALUE_DIFFERENCE,
                ConflictResolution(ResolutionState.UNRESOLVED),
            )

    def test_conflicting_values_retain_provenance_and_require_explicit_resolution(self) -> None:
        bpk = SourceProvenance(SourceKind.OFFICIAL_CATALOG_PAGE, CATALOG.reference, NOW, source_authority="BPK")
        jdih = SourceProvenance(SourceKind.OFFICIAL_JDIH_PAGE, "https://example.invalid/jdih", NOW, source_authority="JDIH")
        values = (
            OfficialValue("Judul A", "judul a", "Judul A", bpk),
            OfficialValue("Judul B", "judul b", "Judul B", jdih),
        )
        unresolved = VerifiedValue(
            None,
            None,
            None,
            FieldState.CONFLICTING_SOURCES,
            conflicting_values=values,
            conflict_kind=ConflictKind.SOURCE_VALUE_DIFFERENCE,
            resolution=ConflictResolution(ResolutionState.UNRESOLVED),
        )
        self.assertIsNone(unresolved.display_value)
        with self.assertRaisesRegex(ValueError, "resolved_conflict_requires_decision"):
            ConflictResolution(ResolutionState.RESOLVED, 1)
        resolved = VerifiedValue(
            "Judul B",
            "judul b",
            "Judul B",
            FieldState.CONFLICTING_SOURCES,
            jdih,
            values,
            ConflictKind.SOURCE_VALUE_DIFFERENCE,
            ConflictResolution(ResolutionState.RESOLVED, 1, "Pilih naskah resmi", "Judul penetapan"),
        )
        self.assertEqual(resolved.provenance, jdih)

    def test_catalog_facts_cannot_claim_pdf_grounding(self) -> None:
        with self.assertRaisesRegex(ValueError, "non_pdf_source_has_pdf_grounding"):
            SourceProvenance(SourceKind.OFFICIAL_CATALOG_PAGE, CATALOG.reference, NOW, page_number=1, selector="Berlaku")

    def test_directional_relations_have_explicit_inverse_endpoints(self) -> None:
        relation = DocumentRelation(RelationKind.AMENDS, "modifier", "base", CATALOG)
        self.assertEqual(relation.inverse(), DocumentRelation(RelationKind.AMENDED_BY, "base", "modifier", CATALOG))

    def test_status_scope_is_explicit_and_closed(self) -> None:
        status = StatusAssertion(VerifiedValue("Berlaku", "berlaku", "Berlaku", FieldState.VERIFIED, CATALOG), NOW, "parent_record")
        self.assertEqual(status.scope, "parent_record")
        with self.assertRaisesRegex(ValueError, "invalid_status_scope"):
            StatusAssertion(status.status, NOW, "unknown")

    def test_provision_effect_requires_exact_pdf_target_and_selector(self) -> None:
        effect = ProvisionEffect(RelationKind.AMENDS, "modifier", "base", "Pasal 1", "Mengubah Pasal 1.", PDF)
        self.assertEqual(effect.exact_target, "Pasal 1")
        with self.assertRaisesRegex(ValueError, "provision_effect_requires_official_pdf"):
            ProvisionEffect(RelationKind.AMENDS, "modifier", "base", "Pasal 1", "Mengubah Pasal 1.", CATALOG)
        with self.assertRaisesRegex(ValueError, "provision_effect_selector_mismatch"):
            ProvisionEffect(RelationKind.AMENDS, "modifier", "base", "Pasal 1", "Teks lain.", PDF)
        with self.assertRaisesRegex(ValueError, "invalid_provision_effect_operation"):
            ProvisionEffect(RelationKind.AMENDS, "modifier", "base", "Pasal 1", "Mengubah Pasal 1.", PDF, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
