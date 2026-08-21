"""One deterministic projection for the common public answer envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tjipto.retrieval.sufficiency import EvidenceRequirement


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
    document_relations: tuple[dict, ...] | None = None
    article_amendment_relations: tuple[dict, ...] | None = None
    relation_support: tuple[dict, ...] | None = None


def project_response(base: dict[str, Any], decision: AnswerDecision) -> dict[str, Any]:
    """Preserve route diagnostics while projecting one complete answer shape."""
    response = base | decision.__dict__
    if decision.reason_code is None:
        response.pop("reason_code", None)
    for key in ("document_relations", "article_amendment_relations", "relation_support"):
        if response[key] is None:
            response.pop(key)
    return response


def compose_research_answer(
    evidence: tuple[dict, ...],
    evidence_set,
    requirements: tuple[EvidenceRequirement, ...],
    assessment,
) -> str:
    """Compose only requirement-assigned, source-backed findings."""
    by_id = {str(row.get("evidence_id")): row for row in evidence}
    requirement_by_id = {row.requirement_id: row for row in requirements}
    findings: list[str] = []
    labels: list[str] = []
    for requirement_id, support_ids in evidence_set.assignments:
        requirement = requirement_by_id.get(requirement_id)
        heading = requirement.description if requirement and requirement.description else requirement_id.replace("_", " ")
        for row in (by_id[support_id] for support_id in support_ids if support_id in by_id):
            label = str(row.get("label") or row.get("citation") or "Ketentuan")
            quote = " ".join(str(row.get("quoted_text") or row.get("display_text") or "").split())
            labels.append(label)
            findings.append(f"{heading}: {label} — {quote}")
    unique_labels = tuple(dict.fromkeys(label for label in labels if label))
    direct = (
        f"Dukungan hukum terverifikasi yang relevan terdapat pada {', '.join(unique_labels)}."
        if unique_labels
        else "Dukungan hukum terverifikasi tersedia untuk kebutuhan yang dipenuhi."
    )
    roles = tuple(dict.fromkeys(str(row.get("source_role") or "") for row in evidence))
    qualification = (
        " Sumber yang digunakan bersifat historis dan tidak diperlakukan sebagai naskah konsolidasi saat ini."
        if roles and all(role != "current_consolidated" for role in roles)
        else ""
    )
    limitation = (
        f" Keterbatasan: dukungan untuk {', '.join(assessment.missing_requirement_ids)} belum terverifikasi."
        if assessment is not None and assessment.missing_requirement_ids
        else ""
    )
    return "\n\n".join((direct + qualification, *findings)) + limitation
