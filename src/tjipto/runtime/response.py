"""One deterministic projection for the common public answer envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerDecision:
    status: str
    route: str
    answer_type: str
    answer: str
    context_pack: dict[str, Any]
    evidence: tuple[dict, ...] = ()
    citations: tuple[dict, ...] = ()
    final_citations: tuple[dict, ...] = ()
    historical_citations: tuple[dict, ...] = ()
    metadata_support: tuple[dict, ...] = ()
    structural_support: tuple[dict, ...] = ()
    trace_support: tuple[dict, ...] = ()
    viewer_refs: tuple[dict, ...] = ()
    metadata_facts: tuple[dict, ...] = ()
    legal_relations: tuple[dict, ...] = ()
    answer_scope: str = "insufficient_evidence"
    warnings: tuple[str, ...] = ()
    insufficient_reasons: tuple[str, ...] = ()
    reason_code: str | None = None
    claim_support: tuple[dict, ...] = ()


def project_response(base: dict[str, Any], decision: AnswerDecision) -> dict[str, Any]:
    """Preserve route diagnostics while projecting one complete answer shape."""
    response = base | decision.__dict__
    if decision.reason_code is None:
        response.pop("reason_code", None)
    return response
