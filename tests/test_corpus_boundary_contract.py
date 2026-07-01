from __future__ import annotations

from pathlib import Path
import re
import unittest

from tjipto.corpora.uud import parser


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
        self.assertEqual(parser.parse_uud_bab_reference("BAB XA"), "BAB XA")
        self.assertEqual(parser.parse_uud_pasal_reference("ayat (1) Pasal 28"), "Pasal 28")
        self.assertEqual(parser.parse_uud_ayat_reference("Pasal 28 ayat (1)"), "(1)")
        self.assertEqual(
            parser.normalize_uud_query_reference("pasal 28 (1)"),
            "Pasal 28 ayat (1)",
        )

    def test_runtime_does_not_parse_source_metadata_from_id_shape(self) -> None:
        source = (ROOT / "src/tjipto/runtime/service.py").read_text(encoding="utf-8")
        self.assertNotIn('source_document_id") or "").split("::")', source)
        self.assertNotIn("source_document_id.split", source)


if __name__ == "__main__":
    unittest.main()
