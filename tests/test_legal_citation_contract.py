from __future__ import annotations

from dataclasses import replace
import unittest

from tjipto.contracts.legal_information import CitationUnit
from tjipto.evidence.legal_citation import FootnoteBook, IndonesianLegalCitationProfile


def unit(key: str, provision: str | None = None) -> CitationUnit:
    return CitationUnit(
        key,
        "Peraturan Presiden",
        "98",
        "2020",
        "Gaji dan Tunjangan Pegawai Pemerintah dengan Perjanjian Kerja",
        "Lembaran Negara Republik Indonesia Tahun 2020 Nomor 218",
        provision,
        2,
        "https://peraturan.bpk.go.id/Details/147306/perpres-no",
        "official_pdf",
        True,
    )


class LegalCitationContractTest(unittest.TestCase):
    def test_rendering_is_typed_deterministic_and_uses_short_subsequent_form(self) -> None:
        profile = IndonesianLegalCitationProfile()
        citation = unit("evidence-1", "Pasal 1")
        self.assertEqual(profile.full(citation), profile.full(citation))
        book = FootnoteBook(profile)
        first = book.cite(citation)
        second = book.cite(citation)
        self.assertEqual((first[0], second[0]), (1, 2))
        self.assertNotEqual(first[1], second[1])
        self.assertNotIn("teks bebas", first[1])
        self.assertIn("Peraturan Presiden Nomor 98 Tahun 2020", first[1])

    def test_footnote_numbers_and_deduplication_are_stable(self) -> None:
        book = FootnoteBook()
        first = unit("evidence-1", "Pasal 1")
        second = unit("evidence-2", "Pasal 2")
        self.assertEqual(book.footnote((first, second))[0], 1)
        self.assertEqual(book.cite(second)[0], 2)
        self.assertEqual(book.cite(first)[0], 3)

    def test_constitution_identity_has_no_fabricated_number_or_duplicate_title(self) -> None:
        citation = replace(
            unit("uud", "Pasal 7"),
            document_type="Undang-Undang Dasar",
            number=None,
            year="1945",
            official_title="Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
        )
        rendered = IndonesianLegalCitationProfile().full(citation)
        self.assertEqual(rendered.count("Undang-Undang Dasar"), 1)
        self.assertNotIn("Nomor 1945", rendered)

    def test_multisource_footnote_consumes_one_number_and_deduplicates(self) -> None:
        book = FootnoteBook()
        first = unit("evidence-1", "Pasal 1")
        other = replace(
            unit("evidence-2", "Pasal 2"),
            document_type="Undang-Undang",
            number="1",
            year="2024",
            official_title="Contoh",
            official_url="https://example.invalid/uu-1-2024",
        )
        number, text = book.footnote((other, first, other))
        self.assertEqual(number, 1)
        self.assertEqual(text.count("Undang-Undang Nomor 1 Tahun 2024"), 1)
        self.assertEqual(book.cite(first)[0], 2)

    def test_multisource_order_is_deterministic_and_history_controls_short_form(self) -> None:
        first = unit("evidence-1", "Pasal 1")
        other = replace(first, evidence_key="evidence-2", document_type="Undang-Undang", official_title="Contoh")
        left = FootnoteBook()
        right = FootnoteBook()
        self.assertEqual(left.footnote((first, other))[1], right.footnote((other, first))[1])
        same_document = replace(first, evidence_key="evidence-3", official_url="https://example.invalid/alternate")
        self.assertEqual(left.cite(same_document), (2, left.profile.short(same_document)))


if __name__ == "__main__":
    unittest.main()
