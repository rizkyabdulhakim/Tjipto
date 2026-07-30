from __future__ import annotations

from datetime import datetime, timezone
import unittest

from tjipto.contracts.legal_information import (
    DocumentRelation,
    FieldState,
    ProvisionEffect,
    RelationKind,
    SourceKind,
    SourceProvenance,
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
    def test_verified_values_preserve_source_normalized_display_and_provenance(self) -> None:
        value = VerifiedValue("Berlaku ", "applicable", "Berlaku", FieldState.VERIFIED, CATALOG)
        self.assertEqual((value.source_value, value.normalized_value, value.display_value), ("Berlaku ", "applicable", "Berlaku"))
        with self.assertRaisesRegex(ValueError, "verified_value_requires_source_and_provenance"):
            VerifiedValue("Berlaku", "applicable", "Berlaku", FieldState.VERIFIED)

    def test_conflicts_cannot_be_silently_collapsed(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting_value_collapsed"):
            VerifiedValue(None, "applicable", None, FieldState.CONFLICTING_SOURCES)

    def test_catalog_facts_cannot_claim_pdf_grounding(self) -> None:
        with self.assertRaisesRegex(ValueError, "non_pdf_source_has_pdf_grounding"):
            SourceProvenance(SourceKind.OFFICIAL_CATALOG_PAGE, CATALOG.reference, NOW, page_number=1, selector="Berlaku")

    def test_directional_relations_have_explicit_inverse_endpoints(self) -> None:
        relation = DocumentRelation(RelationKind.AMENDS, "modifier", "base", CATALOG)
        self.assertEqual(relation.inverse(), DocumentRelation(RelationKind.AMENDED_BY, "base", "modifier", CATALOG))

    def test_provision_effect_requires_exact_pdf_target_and_selector(self) -> None:
        effect = ProvisionEffect(RelationKind.AMENDS, "modifier", "base", "Pasal 1", "Mengubah Pasal 1.", PDF)
        self.assertEqual(effect.exact_target, "Pasal 1")
        with self.assertRaisesRegex(ValueError, "provision_effect_requires_official_pdf"):
            ProvisionEffect(RelationKind.AMENDS, "modifier", "base", "Pasal 1", "Mengubah Pasal 1.", CATALOG)
        with self.assertRaisesRegex(ValueError, "provision_effect_selector_mismatch"):
            ProvisionEffect(RelationKind.AMENDS, "modifier", "base", "Pasal 1", "Teks lain.", PDF)


if __name__ == "__main__":
    unittest.main()
