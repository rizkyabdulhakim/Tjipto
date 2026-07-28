from __future__ import annotations

import unittest

from tjipto.runtime.response import AnswerDecision, project_response


class ResponseProjectionContractTest(unittest.TestCase):
    def test_projection_keeps_route_diagnostics_and_complete_empty_support(self) -> None:
        response = project_response(
            {"intent": "exact_citation", "reason": "citation_not_found"},
            AnswerDecision("insufficient_evidence", "legal_reference", "none", "insufficient", {"answer_evidence": ()}),
        )
        self.assertEqual((response["status"], response["route"], response["intent"]), ("insufficient_evidence", "legal_reference", "exact_citation"))
        self.assertEqual((response["evidence"], response["citations"], response["viewer_refs"]), ((), (), ()))
        self.assertNotIn("reason_code", response)


if __name__ == "__main__":
    unittest.main()
