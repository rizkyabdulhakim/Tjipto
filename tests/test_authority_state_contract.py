from __future__ import annotations

import unittest

from tjipto.contracts.authority import authority_decision, authority_state_error


class AuthorityStateContractTest(unittest.TestCase):
    def test_allowed_states(self) -> None:
        cases = (
            ("normative_legal_text", True, True, "exact", True),
            ("normative_legal_text", False, False, "exact", True),
            ("structural_context", False, False, "not_applicable", False),
            ("endpoint_provenance", False, False, "not_applicable", False),
            ("instrument_provenance", False, False, "not_applicable", False),
            ("source_anomaly_trace", False, False, "not_applicable", False),
            ("page_only", False, False, "page_only", True),
            ("rejected", False, False, "rejected", False),
            ("nonlegal", False, False, "not_applicable", False),
        )
        for kind, citable, final, exactness, evidence in cases:
            with self.subTest(kind=kind):
                self.assertIsNone(
                    authority_state_error(
                        authority_kind=kind,
                        citable=citable,
                        citation_final=final,
                        exactness=exactness,
                        evidence_exists=evidence,
                        reason_code="test",
                    )
                )
                self.assertEqual(
                    authority_decision(
                        authority_kind=kind,
                        citable=citable,
                        citation_final=final,
                        exactness=exactness,
                        evidence_exists=evidence,
                        reason_code="test",
                    )["citation_final"],
                    final,
                )

    def test_forbidden_states(self) -> None:
        cases = (
            ("unknown", False, False, "not_applicable", False, "authority_kind_unknown"),
            ("trace", False, False, "not_applicable", False, "authority_kind_unknown"),
            ("structural_context", False, True, "exact", True, "authority_nonlegal_final"),
            ("normative_legal_text", False, True, "exact", True, "authority_final_without_exact_evidence"),
            ("normative_legal_text", True, True, "page_only", True, "authority_final_without_exact_evidence"),
            ("normative_legal_text", True, True, "exact", False, "authority_final_without_exact_evidence"),
            ("metadata", True, False, "exact", True, "authority_citable_without_exact_evidence"),
            ("nonlegal", False, False, "unknown", False, "authority_exactness_unknown"),
        )
        for kind, citable, final, exactness, evidence, expected in cases:
            with self.subTest(kind=kind, expected=expected):
                self.assertEqual(
                    authority_state_error(
                        authority_kind=kind,
                        citable=citable,
                        citation_final=final,
                        exactness=exactness,
                        evidence_exists=evidence,
                        reason_code="test",
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
