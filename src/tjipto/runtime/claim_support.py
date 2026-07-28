"""Deterministic, clause-grounded substantive-claim gate."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.runtime.query_semantics import PropositionClaim, QuerySemantics


@dataclass(frozen=True)
class ClaimSupport:
    claim_id: str
    claim_text: str
    status: str
    support_evidence_ids: tuple[str, ...]
    source_role: str | None
    temporal_context: str | None
    validation_method: str
    reason_code: str | None
    predicate: str
    polarity: str
    modality: str
    text_span_ids: tuple[str, ...] = ()

    def public(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status,
            "support_evidence_ids": self.support_evidence_ids,
            "source_role": self.source_role,
            "temporal_context": self.temporal_context,
            "validation_method": self.validation_method,
            "reason_code": self.reason_code,
            "predicate": self.predicate,
            "polarity": self.polarity,
            "modality": self.modality,
            "text_span_ids": self.text_span_ids,
        }


@dataclass(frozen=True)
class ClauseProposition:
    """A BBox-backed clause is the smallest authority used for a claim."""

    subject: str | None
    predicate: str
    object: str
    polarity: str
    modality: str
    conditions: tuple[str, ...]
    exceptions: tuple[str, ...]
    source_role: str | None
    temporal_context: str | None
    evidence_id: str
    text_span_ids: tuple[str, ...]


def verify_claims(semantics: QuerySemantics, evidence: tuple[dict, ...], store) -> tuple[ClaimSupport, ...]:
    if semantics.requested_function != "proposition_verification":
        return ()
    claim = semantics.requested_proposition
    if claim is None:
        return ()
    clauses = _grounded_clauses(store, evidence)
    support = tuple(clause for clause in clauses if _supports(claim, clause))
    first = support[0] if support else (clauses[0] if clauses else None)
    # Absence is never an opposite proposition.  A future corpus parser may
    # emit an explicit opposite ClauseProposition; this conservative baseline
    # fails closed until it does.
    status = "supported" if support else "insufficient"
    support_ids = tuple(clause.evidence_id for clause in support)
    return (
        ClaimSupport(
            claim_id="proposition:0",
            claim_text=claim.object,
            status=status,
            support_evidence_ids=support_ids,
            source_role=first.source_role if first else claim.source_role,
            temporal_context=first.temporal_context if first else claim.temporal_context,
            validation_method="bbox_backed_clause_exact_text",
            reason_code=None if support else "claim_support_insufficient",
            predicate=claim.predicate,
            polarity=claim.polarity,
            modality=claim.modality,
            text_span_ids=support[0].text_span_ids if support else (),
        ),
    )


def all_supported(claims: tuple[ClaimSupport, ...]) -> bool:
    return all(claim.status == "supported" for claim in claims)


def _grounded_clauses(store, evidence: tuple[dict, ...]) -> tuple[ClauseProposition, ...]:
    units = {row.get("legal_unit_id") for row in evidence if row.get("legal_unit_id")}
    children = {
        row["legal_unit_id"]
        for row in store.legal_units
        if row.get("parent_legal_unit_id") in units and row.get("unit_type") == "ayat_record"
    }
    target_units = children or units
    out = []
    for unit in store.legal_units:
        if unit.get("legal_unit_id") not in target_units:
            continue
        spans = tuple(unit.get("text_span_ids") or ())
        evidence_id = next((str(item) for item in unit.get("evidence_ids") or () if store.get(item)), None)
        record = store.get(evidence_id) if evidence_id else None
        if not isinstance(evidence_id, str) or not record or not spans:
            continue
        for segment_spans, text in _sentence_segments(store, spans):
            if not store.exact_bboxes_for_text_spans(segment_spans):
                continue
            out.append(
                ClauseProposition(
                    subject=None,
                    predicate="textual",
                    object=text,
                    polarity="positive",
                    modality="textual",
                    conditions=(),
                    exceptions=(),
                    source_role=record.get("source_role"),
                    temporal_context=record.get("temporal_context"),
                    evidence_id=evidence_id,
                    text_span_ids=segment_spans,
                )
            )
    return tuple(out)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _sentence_segments(store, spans: tuple[str, ...]) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Keep adjacent selectors together only through one source sentence."""
    by_id = {str(row.get("text_span_id")): row for row in store.page_text_spans}
    segments: list[tuple[tuple[str, ...], str]] = []
    current_ids: list[str] = []
    current_text: list[str] = []
    for span_id in spans:
        span = by_id.get(span_id)
        if span is None:
            return ()
        text = str(span.get("text") or "")
        for sentence_part in re.findall(r"[^.!?]+(?:[.!?]|$)", text):
            current_ids.append(span_id)
            current_text.append(sentence_part)
            if re.search(r"[.!?]\s*$", sentence_part):
                normalized = _normalized(" ".join(current_text))
                if normalized:
                    segments.append((tuple(dict.fromkeys(current_ids)), normalized))
                current_ids, current_text = [], []
    if current_ids:
        normalized = _normalized(" ".join(current_text))
        if normalized:
            segments.append((tuple(current_ids), normalized))
    return tuple(segments)


def _supports(claim: PropositionClaim, clause: ClauseProposition) -> bool:
    # Normative parsing is deliberately not inferred from article words.  Until
    # a strategy supplies a compatible structured clause proposition, only an
    # exact textual assertion may be published.
    return (
        claim.modality == "textual"
        and claim.polarity == "positive"
        and _normalized(claim.object) in clause.object
    )
