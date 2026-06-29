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

    def test_builder_seed_dependency_is_isolated_as_compatibility_bridge(self) -> None:
        source = (ROOT / "src/tjipto/corpora/uud_artifact_baseline.py").read_text(encoding="utf-8")
        seed = (ROOT / "src/tjipto/corpora/uud/compatibility_seed.py").read_text(encoding="utf-8")
        self.assertIn("load_compatibility_seed(final_dir)", source)
        self.assertNotIn("read_jsonl(final_dir", source)
        self.assertIn("compatibility bridge", seed)


if __name__ == "__main__":
    unittest.main()
