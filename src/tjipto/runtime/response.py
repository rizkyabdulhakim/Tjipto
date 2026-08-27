"""One deterministic projection for the common public answer envelope."""

from __future__ import annotations

from dataclasses import dataclass
import re
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
    grouped_findings: dict[str, list[str]] = {}
    for requirement_id, support_ids in evidence_set.assignments:
        requirement = requirement_by_id.get(requirement_id)
        heading = requirement.description if requirement and requirement.description else requirement_id.replace("_", " ")
        for row in (by_id[support_id] for support_id in support_ids if support_id in by_id):
            quote = " ".join(str(row.get("quoted_text") or row.get("display_text") or "").split())
            citation = str(row.get("citation") or "").strip()
            if citation and quote.casefold().startswith(citation.casefold()):
                quote = quote[len(citation):].lstrip(" :.-")
            if requirement_id.startswith("instrument_"):
                scope = re.search(r"\bmengubah\b.*?(?=\s+sehingga\s+selengkapnya)", quote, re.IGNORECASE)
                quote = scope.group(0).rstrip(" .:") if scope else quote
            if requirement_id.startswith("source_occurrence_"):
                document = _source_occurrence_document(requirement, heading)
                provision = _provision_reference(row, citation)
                location = f" pada {provision}" if provision else ""
                findings.append(f"Dalam {document}, ketentuan tersebut tercantum{location}: {quote}")
            else:
                # Several analysis requirements intentionally share one
                # configured heading (issue provisions).  Emit that heading
                # once while retaining every source-backed quote, avoiding a
                # repetitive template without inventing connective text.
                grouped_findings.setdefault(heading, []).append(quote)
    findings.extend(
        f"{heading}: {' '.join(quotes)}"
        for heading, quotes in grouped_findings.items()
    )
    roles = tuple(dict.fromkeys(str(row.get("source_role") or "") for row in evidence))
    qualification = (
        "Sumber yang digunakan merupakan naskah historis, bukan naskah konsolidasi yang berlaku saat ini."
        if roles and all(role != "current_consolidated" for role in roles)
        else ""
    )
    missing = tuple(
        requirement_by_id[requirement_id].description
        for requirement_id in (assessment.missing_requirement_ids if assessment is not None else ())
        if requirement_id in requirement_by_id
    )
    missing_ids = assessment.missing_requirement_ids if assessment is not None else ()
    if missing and all(requirement_id.startswith("source_occurrence_") for requirement_id in missing_ids):
        documents = tuple(
            _source_occurrence_document(requirement_by_id.get(requirement_id), requirement_id)
            for requirement_id in missing_ids
            if requirement_id in requirement_by_id
        )
        limitation = f"Belum ditemukan dukungan terverifikasi pada {', '.join(documents)}."
    else:
        limitation = f"Keterbatasan: dukungan untuk {', '.join(missing)} belum terverifikasi." if missing else ""
    paragraphs = tuple((*findings, qualification, limitation))
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _provision_reference(row: dict, fallback: str) -> str:
    hierarchy = tuple(str(value) for value in row.get("hierarchy") or () if value)
    for index, value in enumerate(hierarchy):
        if value.casefold().startswith("pasal "):
            return " / ".join(hierarchy[index:])
    return fallback


def _source_occurrence_document(requirement: EvidenceRequirement | None, fallback: str) -> str:
    if requirement is None:
        return fallback
    description = requirement.description or fallback
    prefix = f"{requirement.retrieval_query} dalam " if requirement.retrieval_query else ""
    return description[len(prefix):] if prefix and description.startswith(prefix) else description
