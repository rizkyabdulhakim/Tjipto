"""Optional, untrusted wording adapters for server-rendered answers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from builtins import object as builtin_object
from typing import Protocol
from urllib.parse import urlparse


class WordingProvider(Protocol):
    def propose(self, deterministic_answer: str) -> dict[str, object] | None: ...


@dataclass(frozen=True)
class AnswerProposition:
    """Server-owned factual slots exposed to the optional wording adapter.

    The adapter can choose and order these facts, but it never receives an
    authority to edit their slots.  Empty optional fields mean that the
    evidence row did not prove that dimension; they are not inferred here.
    """

    fact_id: str
    support_ids: tuple[str, ...]
    subject: str | None
    predicate: str | None
    object: str | None
    legal_references: tuple[str, ...] = ()
    modality: str | None = None
    polarity: str | None = None
    numbers: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    source_role: str | None = None
    temporal_scope: str | None = None

    @property
    def claim_id(self) -> str:
        return self.fact_id

    def public(self) -> dict[str, builtin_object]:
        value = asdict(self)
        value["claim_id"] = value["fact_id"]
        return value


@dataclass(frozen=True)
class AnswerFactPlan:
    """Immutable, request-local facts that may be selected for wording."""

    facts: tuple[AnswerProposition, ...]

    def public(self) -> tuple[dict[str, object], ...]:
        return tuple(fact.public() for fact in self.facts)


@dataclass(frozen=True)
class VerifiedClaimSet:
    """Immutable, server-owned claims exposed to the optional wording layer."""

    claims: tuple[AnswerProposition, ...]

    def public(self) -> tuple[dict[str, object], ...]:
        return tuple(claim.public() for claim in self.claims)


def build_answer_fact_plan(evidence: tuple[dict, ...], fallback: str) -> AnswerFactPlan:
    """Project only source-backed fields into the wording boundary."""
    facts = [
        AnswerProposition(
            fact_id="deterministic_answer",
            support_ids=(),
            subject=None,
            predicate=None,
            object=fallback,
        )
    ]
    for row in evidence:
        evidence_id = str(row.get("evidence_id") or "")
        quote = str(row.get("quoted_text") or row.get("display_text") or "").strip()
        if not evidence_id or not quote:
            continue
        references = tuple(
            str(value)
            for value in (row.get("citation"), row.get("target_citation"))
            if isinstance(value, str) and value.strip()
        )
        facts.append(
            AnswerProposition(
                fact_id=f"support:{evidence_id}",
                support_ids=(evidence_id,),
                subject=row.get("subject") if isinstance(row.get("subject"), str) else None,
                predicate=row.get("predicate") if isinstance(row.get("predicate"), str) else None,
                object=quote,
                legal_references=tuple(dict.fromkeys(references)),
                modality=row.get("modality") if isinstance(row.get("modality"), str) else None,
                polarity=row.get("polarity") if isinstance(row.get("polarity"), str) else None,
                numbers=_string_values(row.get("numbers")),
                dates=_string_values(row.get("dates")),
                source_role=row.get("source_role") if isinstance(row.get("source_role"), str) else None,
                temporal_scope=row.get("temporal_context") if isinstance(row.get("temporal_context"), str) else None,
            )
        )
    return AnswerFactPlan(tuple(facts))


def build_verified_claim_set(evidence: tuple[dict, ...]) -> VerifiedClaimSet:
    """Project only verified evidence rows into immutable claim slots."""
    plan = build_answer_fact_plan(evidence, "")
    return VerifiedClaimSet(tuple(fact for fact in plan.facts if fact.fact_id != "deterministic_answer"))


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, int, float)):
        return (str(value),)
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value if isinstance(item, (str, int, float)))
    return ()


def wording_enabled_from_environment() -> bool:
    return os.environ.get("TJIPTO_EXTERNAL_WORDING", "").strip().casefold() == "enabled"


def wording_provider_from_environment() -> WordingProvider | None:
    """Build an opted-in adapter; invalid optional configuration is harmless."""
    if not wording_enabled_from_environment():
        return None
    provider = os.environ.get("TJIPTO_WORDING_PROVIDER", "").strip().casefold()
    api_key = os.environ.get("TJIPTO_WORDING_API_KEY", "").strip()
    model = os.environ.get("TJIPTO_WORDING_MODEL", "").strip()
    base_url = os.environ.get("TJIPTO_WORDING_BASE_URL", "").strip()
    if not provider or not api_key or not model:
        return None
    try:
        timeout = max(1.0, float(os.environ.get("TJIPTO_WORDING_TIMEOUT_SECONDS", "12")))
    except ValueError:
        return None
    if provider == "gemini":
        from tjipto.runtime.gemini import GeminiAnswerProvider

        endpoint = base_url or "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        if not _https(endpoint):
            return None
        return GeminiAnswerProvider(api_key, model=model, endpoint=endpoint, timeout=timeout)
    if provider == "openai_compatible":
        from tjipto.runtime.openai_compatible import OpenAICompatibleWordingProvider

        endpoint = base_url.rstrip("/") + "/chat/completions" if base_url else ""
        if not _https(endpoint):
            return None
        return OpenAICompatibleWordingProvider(api_key, model=model, endpoint=endpoint, timeout=timeout)
    return None


def valid_proposal(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    presentation = value.get("presentation")
    references = value.get("referenced_fact_ids")
    if presentation in {"direct", "grounded"} and isinstance(references, list) and all(isinstance(item, str) for item in references):
        return {"presentation": presentation, "referenced_fact_ids": tuple(references)}
    sentences = value.get("sentences")
    if set(value) != {"sentences"} or not isinstance(sentences, list) or not sentences:
        return None
    normalized = []
    for sentence in sentences:
        if not isinstance(sentence, dict):
            return None
        if set(sentence) == {"style", "referenced_fact_ids"}:
            style = sentence.get("style")
            refs = sentence.get("referenced_fact_ids")
            if style not in {"direct", "grounded"} or not isinstance(refs, list) or not refs or not all(isinstance(item, str) for item in refs):
                return None
            normalized.append({"style": style, "referenced_fact_ids": tuple(refs)})
            continue
        if set(sentence) != {"text", "claim_ids"}:
            return None
        text = sentence.get("text")
        claim_ids = sentence.get("claim_ids")
        if not isinstance(text, str) or not text.strip() or not isinstance(claim_ids, list) or not claim_ids or not all(
            isinstance(item, str) and item.strip() for item in claim_ids
        ):
            return None
        normalized.append({"text": text.strip(), "claim_ids": tuple(claim_ids)})
    return {"sentences": tuple(normalized)}


def _https(value: str) -> bool:
    parsed = urlparse(value.format(model="model"))
    return parsed.scheme == "https" and bool(parsed.netloc)
