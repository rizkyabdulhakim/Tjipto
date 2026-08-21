"""Optional, untrusted wording adapters for server-rendered answers."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from builtins import object as builtin_object
from typing import Protocol
from urllib.parse import urlparse

from tjipto.core.external_llm import external_llm_config
from tjipto.corpora.intent_config import normalize_intent_text


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
    verified_span: str | None = None

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
                verified_span=quote,
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
    config = external_llm_config("WORDING")
    if config is None:
        return None
    if config.provider == "gemini":
        from tjipto.runtime.gemini import GeminiAnswerProvider

        endpoint = config.base_url or "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        if not _https(endpoint):
            return None
        return GeminiAnswerProvider(config.api_key, model=config.model, endpoint=endpoint, timeout=config.timeout)
    if config.provider == "openai_compatible":
        from tjipto.runtime.openai_compatible import OpenAICompatibleWordingProvider

        endpoint = config.base_url + "/chat/completions" if config.base_url else ""
        if not _https(endpoint):
            return None
        return OpenAICompatibleWordingProvider(config.api_key, model=config.model, endpoint=endpoint, timeout=config.timeout)
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


def render_wording(
    proposal: object,
    fallback: str,
    facts: dict[str, str] | None = None,
    verified_claims: VerifiedClaimSet | None = None,
) -> str:
    """Render server-owned fact slots; an external model never publishes unchecked prose."""
    if isinstance(proposal, dict) and set(proposal) == {"sentences"}:
        sentences = proposal.get("sentences")
        if verified_claims is not None and sentences and all(
            isinstance(sentence, dict) and set(sentence) == {"text", "claim_ids"}
            for sentence in sentences
        ):
            return _render_verified_synthesis(sentences, fallback, verified_claims)
        approved = facts or {"deterministic_answer": fallback}
        if not isinstance(sentences, tuple) or not sentences:
            return fallback
        rendered: list[str] = []
        for sentence in sentences:
            if not isinstance(sentence, dict) or set(sentence) != {"style", "referenced_fact_ids"}:
                return fallback
            refs = sentence.get("referenced_fact_ids")
            style = sentence.get("style")
            if style not in {"direct", "grounded"} or not isinstance(refs, tuple) or not refs:
                return fallback
            if len(refs) != len(set(refs)) or not set(refs) <= set(approved):
                return fallback
            body = " ".join(approved[item] for item in refs).strip()
            if not body:
                return fallback
            rendered.append(f"Berdasarkan bukti terverifikasi, {body}" if style == "grounded" else body)
        return " ".join(rendered)
    if not isinstance(proposal, dict) or set(proposal) != {"presentation", "referenced_fact_ids"}:
        return fallback
    references = proposal.get("referenced_fact_ids")
    if not isinstance(references, tuple):
        return fallback
    approved = facts or {"deterministic_answer": fallback}
    if not references or len(references) != len(set(references)) or not set(references) <= set(approved):
        return fallback
    if proposal.get("presentation") == "grounded":
        return f"Berdasarkan bukti terverifikasi, {' '.join(approved[item] for item in references)}"
    if proposal.get("presentation") == "direct":
        return " ".join(approved[item] for item in references)
    return fallback


_SYNTHESIS_CONNECTORS = frozenset(
    "adalah akan bahwa dengan dan dari dalam ini itu karena ke pada sebagai serta untuk yang terdapat menurut dapat dapatnya".split()
)
_UNSUPPORTED_CONCLUSIONS = ("oleh karena itu", "dengan demikian", "berarti", "sehingga", "maka")


def _render_verified_synthesis(sentences, fallback: str, verified_claims: VerifiedClaimSet) -> str:
    claims = {claim.claim_id: claim for claim in verified_claims.claims}
    if not claims:
        return fallback
    used: set[str] = set()
    rendered: list[str] = []
    for sentence in sentences:
        text = str(sentence.get("text") or "").strip()
        claim_ids = tuple(sentence.get("claim_ids") or ())
        if not text or not claim_ids or len(claim_ids) != len(set(claim_ids)) or not set(claim_ids) <= set(claims):
            return fallback
        used.update(claim_ids)
        selected = tuple(claims[claim_id] for claim_id in claim_ids)
        output_tokens = set(re.findall(r"[a-z0-9]+", normalize_intent_text(text)))
        if any(not _has_structured_roles(claim) for claim in selected):
            spans = tuple(
                normalize_intent_text(claim.verified_span or "")
                for claim in selected
                if not _has_structured_roles(claim)
            )
            if any(not span or span not in normalize_intent_text(text) for span in spans):
                return fallback
            allowed = set(re.findall(r"[a-z0-9]+", " ".join(spans))) | _SYNTHESIS_CONNECTORS
            if not output_tokens <= allowed:
                return fallback
        else:
            allowed = set(_tokens_for_claims(selected)) | _SYNTHESIS_CONNECTORS
            if not output_tokens <= allowed:
                return fallback
            if not set(_protected_tokens_for_claims(selected)) <= output_tokens:
                return fallback
        if not _preserves_polarity(selected, text):
            return fallback
        if any(
            phrase in normalize_intent_text(text) and phrase not in " ".join(_claim_texts(selected))
            for phrase in _UNSUPPORTED_CONCLUSIONS
        ):
            return fallback
        if _reverses_subject_object(selected, text) or _scope_drift(selected, text):
            return fallback
        rendered.append(text)
    return " ".join(rendered) if used == set(claims) and rendered else fallback


def _claim_texts(claims) -> tuple[str, ...]:
    values: list[str] = []
    for claim in claims:
        values.extend(
            value
            for value in (
                claim.subject,
                claim.predicate,
                claim.object,
                *claim.legal_references,
                claim.modality,
                *claim.numbers,
                *claim.dates,
            )
            if value
        )
    return tuple(str(value) for value in values)


def _tokens_for_claims(claims) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"[a-z0-9]+", normalize_intent_text(" ".join(_claim_texts(claims))))))


def _protected_tokens_for_claims(claims) -> tuple[str, ...]:
    values: list[str] = []
    for claim in claims:
        values.extend(
            value
            for value in (
                claim.subject,
                claim.predicate,
                claim.object if _has_structured_roles(claim) else None,
                *claim.legal_references,
                claim.modality,
                *claim.numbers,
                *claim.dates,
            )
            if value
        )
    normalized = normalize_intent_text(" ".join(str(value) for value in values))
    return tuple(dict.fromkeys(re.findall(r"[a-z0-9]+", normalized)))


def _has_structured_roles(claim) -> bool:
    return all(
        isinstance(getattr(claim, field, None), str) and getattr(claim, field).strip()
        for field in ("subject", "predicate", "object")
    )


def _scope_drift(claims, text: str) -> bool:
    normalized = normalize_intent_text(text)
    historical_markers = ("historis", "naskah asli", "amandemen", "perubahan pertama", "perubahan kedua", "perubahan ketiga", "perubahan keempat")
    current_markers = ("konsolidasi", "satu naskah", "berlaku saat ini", "saat ini")
    for claim in claims:
        role = normalize_intent_text(claim.source_role or claim.temporal_scope or "")
        if role == "current consolidated" and any(marker in normalized for marker in historical_markers[:2]):
            return True
        if role and role != "current consolidated" and any(marker in normalized for marker in current_markers):
            return True
    return False


def _preserves_polarity(claims, text: str) -> bool:
    output = set(re.findall(r"[a-z0-9]+", normalize_intent_text(text)))
    negations = {"tidak", "bukan", "belum", "tanpa"}
    for claim in claims:
        polarity = normalize_intent_text(claim.polarity or "")
        claim_tokens = set(re.findall(r"[a-z0-9]+", normalize_intent_text(" ".join(_claim_texts((claim,))))))
        if any(marker in polarity for marker in ("negative", "negated", "not")):
            if not output.intersection(negations):
                return False
        elif output.intersection(negations) - claim_tokens:
            return False
    return True


def _reverses_subject_object(claims, text: str) -> bool:
    normalized = normalize_intent_text(text)
    for claim in claims:
        subject = tuple(re.findall(r"[a-z0-9]+", normalize_intent_text(claim.subject or "")))
        object_tokens = tuple(re.findall(r"[a-z0-9]+", normalize_intent_text(claim.object or "")))
        subject_anchor = next((token for token in subject if len(token) > 2), None)
        object_anchor = next((token for token in object_tokens if len(token) > 2 and token != subject_anchor), None)
        if not subject_anchor or not object_anchor:
            continue
        subject_pos = normalized.find(subject_anchor)
        object_pos = normalized.find(object_anchor)
        if subject_pos >= 0 and object_pos >= 0 and subject_pos > object_pos:
            return True
    return False


def _https(value: str) -> bool:
    parsed = urlparse(value.format(model="model"))
    return parsed.scheme == "https" and bool(parsed.netloc)
