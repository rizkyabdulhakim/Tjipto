from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UudBuilderContractTest(unittest.TestCase):
    def test_builder_does_not_strip_old_final_artifact_rows(self) -> None:
        source = (ROOT / "src/tjipto/corpora/uud_artifact_baseline.py").read_text(encoding="utf-8")
        self.assertNotIn("<= 609", source)
        self.assertNotIn('startswith("uud_instrument_final_citation_evidence::")', source)
        self.assertNotIn('startswith("uud_retrieval_unit::uud_instrument_final_citation_evidence::")', source)


if __name__ == "__main__":
    unittest.main()
