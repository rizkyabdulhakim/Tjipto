from __future__ import annotations

from pathlib import Path
import re
import unittest

from tjipto.corpora import parser_dispatch
from tjipto.corpora.uud import parser as uud_parser


ROOT = Path(__file__).resolve().parents[1]


class CorpusBoundaryContractTest(unittest.TestCase):
    def test_generic_layers_do_not_define_uud_legal_regex(self) -> None:
        for rel_path, names in {
            "src/tjipto/retrieval/query.py": ("PASAL_RE", "AYAT_RE", "PASAL_SHORTHAND_AYAT_RE"),
            "src/tjipto/retrieval/structured.py": ("BAB_RE", "PASAL_RE", "AYAT_RE"),
            "src/tjipto/retrieval/relations.py": ("BAB_RE", "PASAL_RE"),
            "src/tjipto/evidence/citation.py": ("PASAL_RE", "AYAT_RE"),
        }.items():
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            for name in names:
                self.assertIsNone(re.search(rf"(?m)^{name}\s*=", source), rel_path)

    def test_uud_parser_owns_legal_reference_helpers(self) -> None:
        self.assertEqual(uud_parser.parse_uud_bab_reference("BAB XA"), "BAB XA")
        self.assertEqual(uud_parser.parse_uud_pasal_reference("ayat (1) Pasal 28"), "Pasal 28")
        self.assertEqual(uud_parser.parse_uud_ayat_reference("Pasal 28 ayat (1)"), "(1)")
        self.assertEqual(
            uud_parser.normalize_uud_query_reference("pasal 28 (1)"),
            "Pasal 28 ayat (1)",
        )

    def test_parser_dispatch_resolves_uud_and_fails_safely(self) -> None:
        parser = parser_dispatch.get_parser("uud")
        self.assertIs(parser.normalize_query_reference, uud_parser.normalize_uud_query_reference)
        self.assertEqual(parser_dispatch.normalize_query_reference("uud", "pasal 28 (1)"), "Pasal 28 ayat (1)")
        self.assertEqual(
            parser_dispatch.parse_legal_reference("uud", "BAB XA Pasal 28 ayat (1)"),
            {"bab": "BAB XA", "pasal": "Pasal 28", "ayat": "(1)"},
        )
        self.assertIn("bab xa", parser_dispatch.label_keys("uud", "BAB X A"))
        with self.assertRaisesRegex(ValueError, "unsupported_corpus_parser:unknown"):
            parser_dispatch.get_parser("unknown")

    def test_parser_dispatch_preserves_all_reference_ranges(self) -> None:
        text = "Pasal19, Pasal\n28C, dan pasal 28G."
        rows = parser_dispatch.parse_legal_references("uud", text)
        self.assertEqual([row["reference"] for row in rows], ["Pasal 19", "Pasal 28C", "Pasal 28G"])
        self.assertEqual([text[int(row["start"]) : int(row["end"])] for row in rows], ["Pasal19", "Pasal\n28C", "pasal 28G"])

    def test_generic_layers_use_parser_dispatch_not_uud_parser(self) -> None:
        for rel_path in (
            "src/tjipto/retrieval/query.py",
            "src/tjipto/retrieval/structured.py",
            "src/tjipto/retrieval/relations.py",
            "src/tjipto/evidence/citation.py",
        ):
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("tjipto.corpora.uud.parser", source, rel_path)

    def test_parser_dispatch_uses_adapter_not_uud_named_calls(self) -> None:
        source = (ROOT / "src/tjipto/corpora/parser_dispatch.py").read_text(encoding="utf-8")
        for leak in ("normalize_uud", "parse_uud", "uud_label"):
            self.assertNotIn(leak, source)

    def test_generic_provenance_has_no_uud_header_logic(self) -> None:
        source = (ROOT / "src/tjipto/corpora/provenance.py").read_text(encoding="utf-8")
        self.assertNotIn("UUD_SATU_NASKAH_HEADER_RE", source)
        self.assertNotIn("_strip_uud_header", source)
        self.assertNotIn("Perubahan Pertama", source)

    def test_runtime_does_not_parse_source_metadata_from_id_shape(self) -> None:
        source = (ROOT / "src/tjipto/runtime/service.py").read_text(encoding="utf-8")
        self.assertNotIn('source_document_id") or "").split("::")', source)
        self.assertNotIn("source_document_id.split", source)


if __name__ == "__main__":
    unittest.main()
