from __future__ import annotations

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
        self.assertEqual((first[0], second[0]), (1, 1))
        self.assertNotEqual(first[1], second[1])
        self.assertNotIn("teks bebas", first[1])

    def test_footnote_numbers_and_deduplication_are_stable(self) -> None:
        book = FootnoteBook()
        first = unit("evidence-1", "Pasal 1")
        second = unit("evidence-2", "Pasal 2")
        self.assertEqual(book.footnote((first, second))[0], 1)
        self.assertEqual(book.cite(second)[0], 2)
        self.assertEqual(book.cite(first)[0], 1)


if __name__ == "__main__":
    unittest.main()
