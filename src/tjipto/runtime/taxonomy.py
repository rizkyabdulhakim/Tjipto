from __future__ import annotations

from enum import StrEnum


class ResultReason(StrEnum):
    ANSWER_READY = "answer_ready"
    CLARIFICATION_REQUIRED = "clarification_required"
    PARTIAL_ANSWER = "partial_answer"
    INSUFFICIENT_SOURCE_EVIDENCE = "insufficient_source_evidence"
    REPRESENTATION_FAILURE = "representation_failure"
    CORPUS_GAP = "corpus_gap"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PUBLICATION_BLOCKED = "publication_blocked"


RESULT_REASON_CODES = frozenset(reason.value for reason in ResultReason)


def validate_reason_code(value: str) -> str:
    if value not in RESULT_REASON_CODES:
        raise ValueError(f"unknown_result_reason:{value}")
    return value


def reason_for_status(status: str, reason: str | None = None) -> str:
    if reason in RESULT_REASON_CODES:
        return reason
    return {
        "answer_ready": ResultReason.ANSWER_READY.value,
        "clarification_required": ResultReason.CLARIFICATION_REQUIRED.value,
        "limited_answer": ResultReason.PARTIAL_ANSWER.value,
        "insufficient_evidence": ResultReason.INSUFFICIENT_SOURCE_EVIDENCE.value,
        "corpus_gap": ResultReason.CORPUS_GAP.value,
        "provider_unavailable": ResultReason.PROVIDER_UNAVAILABLE.value,
    }.get(status, ResultReason.REPRESENTATION_FAILURE.value)


__all__ = ["RESULT_REASON_CODES", "ResultReason", "reason_for_status", "validate_reason_code"]
