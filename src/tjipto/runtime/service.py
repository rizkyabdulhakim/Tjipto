from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from threading import RLock
from collections import Counter, OrderedDict
import unicodedata
from typing import Any
from uuid import uuid4

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text, resolve_instrument_intent
from tjipto.corpora.parser_dispatch import parse_legal_reference
from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.strategy import StrategyRegistry
from tjipto.corpora.verified import CorpusIntegrityError, VerifiedCorpusRepository
from tjipto.evidence.bbox import viewer_overlay_rectangles
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack, validate_answer_candidate
from tjipto.retrieval.bm25 import meaningful_tokens, sparse_index_for_store, tokens
from tjipto.retrieval.candidates import graph_expand
from tjipto.retrieval.metadata import (
    metadata_lookup,
    normalize_filters,
    public_filters,
)
from tjipto.corpora.source_arbitration import resolve_source_scope
from tjipto.retrieval.relations import amendment_relation_target
from tjipto.retrieval.router import route_retrieval
from tjipto.retrieval.research import ResearchIntent, ResearchPlanningProvider, execute_research_rounds
from tjipto.retrieval.sufficiency import EvidenceRequirement, assess_sufficiency, collect_evidence_set
from tjipto.runtime.claim_support import all_supported, verify_claims
from tjipto.runtime.clarification import clarification_decision
from tjipto.runtime.answer_arbitration import document_summary_query, source_document_response
from tjipto.runtime.bookmarks import BookmarkRepository
from tjipto.runtime.query_semantics import interpret_query
from tjipto.runtime.response import AnswerDecision, project_response
from tjipto.runtime.wording import wording_enabled_from_environment, wording_provider_from_environment
from tjipto.runtime.scope_guard import scope_guard_context
from tjipto.runtime.source_text import source_text_response
from tjipto.telemetry import Telemetry
from tjipto.runtime.viewer import _source_status_label, document_viewer_payload, resolve_document_pdf_access, resolve_pdf_access, viewer_payload
from tjipto.catalog import CatalogService


_ANSWER_TEMPLATES = {
    "insufficient": "Bukti tidak cukup atau database belum tersedia dalam korpus terverifikasi saat ini.",
    "legal_relation": "Dukungan relasi hukum berbasis bukti tersedia; sistem tidak menghasilkan kesimpulan hukum.",
    "citation": "Dukungan sitasi berbasis bukti tersedia untuk {citation}; sistem tidak menghasilkan kesimpulan hukum.",
}


def _integrity_failure(corpus_id: str, query: str, error_code: str | None) -> dict:
    unknown = error_code in {"unknown_corpus", "registry_unavailable"}
    route = "unsupported_corpus" if unknown else "corpus_integrity"
    return {
        "status": "unsupported_corpus" if unknown else "corpus_not_ready",
        "route": route,
        "intent": route,
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "reason": error_code or "corpus_load_failure",
        "reason_code": error_code or "corpus_load_failure",
        "readiness": False,
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "context_pack": empty_context_pack(error_code),
        "answer_scope": "insufficient_evidence",
        "answer_type": "none",
        "answer": _ANSWER_TEMPLATES["insufficient"],
    }


def _has_resolved_legal_target(corpus_id: str, query: str, *, config=None) -> bool:
    try:
        return any(parse_legal_reference(corpus_id, query, allow_roman_pasal=True, config=config).values())
    except ValueError:
        return False


def _clarification_candidate_limit(store: EvidenceStore, query: str, limit: int) -> int:
    config = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    terms = tuple(config.get("clarification", {}).get("choice_terms") or ())
    normalized = f" {normalize_intent_text(query)} "
    return len(store.evidence) if any(f" {normalize_intent_text(term)} " in normalized for term in terms) else limit


def _research_candidate_limit(store: EvidenceStore, query: str, limit: int) -> int:
    """Use the corpus-owned bounded research over-fetch budget for sufficiency."""
    research: dict = getattr(getattr(store, "config", None), "setting", lambda *_: {})("research", {}) or {}
    try:
        configured = int(research.get("max_candidates", limit)) if isinstance(research, dict) else limit
    except (TypeError, ValueError):
        configured = limit
    return min(len(store.evidence), max(limit, configured))


def _research_intent_for_ask(
    store: EvidenceStore,
    semantics,
    query: str,
    requirements: tuple[EvidenceRequirement, ...],
) -> ResearchIntent:
    """Derive complexity from the server-owned evidence requirements."""
    if getattr(semantics, "requested_function", "retrieval") not in {"retrieval", "source_discrepancy"}:
        return ResearchIntent()
    config = getattr(store, "config", None)
    research: dict = getattr(config, "setting", lambda *_: {})("research", {}) or {}
    hints = research.get("semantic_hints", {}) if isinstance(research, dict) else {}
    normalized_query = normalize_intent_text(query)
    def configured(name: str) -> bool:
        return bool(
            isinstance(hints, dict)
            and contains_intent_phrase(
                normalized_query,
                tuple(str(value) for value in hints.get(name, ()) if isinstance(value, str)),
            )
        )
    entity_dimensions = tuple(requirement for requirement in requirements if requirement.required_entities)
    # Multiple corpus-backed entities carrying the same semantic dimension are
    # a comparison signal even when the wording is an unseen paraphrase.  The
    # lexical hint remains useful for queries without an explicit dimension,
    # but it is never the source of truth for entity coverage.
    instrument_scopes = _instrument_scope_roles(store, query)
    comparison = (len(entity_dimensions) > 1 or len(instrument_scopes) > 1) and (
        configured("comparison") or configured("authority") or configured("relation")
        or len(instrument_scopes) > 1
    )
    multiple = len(requirements) > 1 or any(requirement.min_supports > 1 for requirement in requirements)
    relation = bool(requirements) and configured("relation")
    decomposition = multiple and (configured("procedure") or comparison)
    return ResearchIntent(
        multiple_supports=multiple,
        comparison=comparison,
        decomposition=decomposition,
        relation_traversal=relation,
        max_variants=max(1, int(research.get("max_variants", 4))) if isinstance(research, dict) else 4,
        max_rounds=max(1, int(research.get("max_rounds", 2))) if isinstance(research, dict) else 2,
    )


def _semantic_scope_covered(
    store: EvidenceStore,
    semantics,
    query: str,
    requirements: tuple[EvidenceRequirement, ...],
) -> bool:
    """Ensure a research plan retains explicit, corpus-backed query dimensions."""
    # Authoritative exact/quotation routes already preserve the parsed legal
    # target and do not create a decomposed research plan.  Their direct
    # resolver is the semantic owner, so entity mentions in the quoted text
    # must not be mistaken for uncovered research dimensions.
    if (
        getattr(semantics, "requested_function", "retrieval") in {"exact_citation", "temporal_quotation"}
        or (
            getattr(semantics, "requested_function", "retrieval") == "proposition_verification"
            and len(tuple(getattr(semantics, "legal_references", ()) or ())) == 1
        )
    ):
        return True
    config = getattr(store, "config", None)
    research: dict = getattr(config, "setting", lambda *_: {})("research", {}) or {}
    normalized = normalize_intent_text(query)
    hints = research.get("semantic_hints", {}) if isinstance(research, dict) else {}

    def hinted(name: str) -> bool:
        values = hints.get(name, ()) if isinstance(hints, dict) else ()
        return contains_intent_phrase(normalized, tuple(str(value) for value in values if isinstance(value, str)))

    entity_labels = set(_research_entities(research, normalized))
    planned_entities = {
        entity
        for requirement in requirements
        for entity in requirement.required_entities
    }
    if len(entity_labels) > 1 and not entity_labels <= planned_entities:
        return False

    instrument_roles = set(_instrument_scope_roles(store, query))
    planned_roles = {
        str(requirement.source_role)
        for requirement in requirements
        if requirement.source_role
    }
    if len(instrument_roles) > 1 and not instrument_roles <= planned_roles:
        return False
    if semantics.source_role and semantics.source_role not in planned_roles and requirements:
        return False

    complex_signal = hinted("comparison") or hinted("procedure") or hinted("relation")
    entity_scope_requirement = any(
        entity_labels <= set(requirement.required_entities)
        for requirement in requirements
    )
    if (
        len(entity_labels) > 1
        and (complex_signal or len(requirements) > 1)
        and len(requirements) < 2
        and not entity_scope_requirement
    ):
        return False
    if complex_signal and not requirements and not _single_support_covers_query(store, query):
        return False
    if semantics.relation_intent and len(entity_labels) > 1:
        covered = all(
            entity in planned_entities
            for entity in entity_labels
        )
        if not covered:
            return False
    return True


def _research_requirements_for_ask(store: EvidenceStore, semantics, query: str) -> tuple[EvidenceRequirement, ...]:
    """Derive typed requirements from corpus-backed semantic dimensions."""
    if getattr(semantics, "requested_function", "retrieval") != "retrieval":
        return ()
    config = getattr(store, "config", None)
    research: dict = getattr(config, "setting", lambda *_: {})("research", {}) or {}
    normalized = normalize_intent_text(query)
    generation = research.get("requirement_generation", {}) if isinstance(research, dict) else {}
    if not isinstance(generation, dict):
        return ()
    delimiters = generation.get("conjunction_delimiters") or ()
    if not isinstance(delimiters, (tuple, list)):
        delimiters = ()
    segments = []
    for delimiter in delimiters:
        if isinstance(delimiter, str):
            segments = [part.strip() for part in normalized.split(f" {normalize_intent_text(delimiter)} ") if part.strip()]
            if len(segments) > 1:
                break
    minimums = generation.get("minimum_supports") or {}
    if not isinstance(minimums, dict):
        minimums = {}
    hints = research.get("semantic_hints", {}) if isinstance(research, dict) else {}
    entities = _research_entities(research, normalized)
    all_entities = tuple(_research_entity_labels(research))
    instrument_roles = _instrument_scope_roles(store, query)

    def hinted(name: str) -> bool:
        values = hints.get(name, ()) if isinstance(hints, dict) else ()
        return contains_intent_phrase(normalized, tuple(str(value) for value in values if isinstance(value, str)))

    support_terms = research.get("support_terms", {}) if isinstance(research, dict) else {}
    authority_terms = tuple(str(value) for value in support_terms.get("authority", ()) if isinstance(value, str))
    procedure_terms = tuple(str(value) for value in support_terms.get("procedure", ()) if isinstance(value, str))
    relation_terms = tuple(str(value) for value in support_terms.get("relation", ()) if isinstance(value, str))
    source_role_labels = intent_config_for(
        getattr(store.config, "structured_strategy", "generic"), store.config
    ).get("source_role_labels", {})
    instrument_scope_terms = tuple(
        str(value).casefold()
        for value in research.get("instrument_scope_terms", ())
        if isinstance(value, str) and value.strip()
    )
    if len(instrument_roles) > 1:
        return tuple(
            EvidenceRequirement(
                f"instrument_{role}",
                description=role,
                retrieval_query=" ".join(
                    (str(source_role_labels.get(role, "")), *instrument_scope_terms)
                ).strip() or query,
                source_role=role,
                temporal_context=role,
                semantic_terms=tuple(meaningful_tokens(str(source_role_labels.get(role, "")))),
                support_terms=instrument_scope_terms,
                authority_kinds=("instrument_provenance", "normative_legal_text"),
            )
            for role in instrument_roles
        )
    if len(entities) > 1 and (hinted("comparison") or (hinted("authority") and len(entities) > 1)):
        return tuple(
            EvidenceRequirement(
                f"entity_{index}",
                description=entity,
                retrieval_query=f"{entity} {' '.join(authority_terms)}".strip(),
                required_entities=(entity,),
                contrast_entities=tuple(value for value in all_entities if value != entity),
                support_terms=authority_terms,
                entity_must_lead=True,
                authority_kinds=("normative_legal_text",),
            )
            for index, entity in enumerate(entities, 1)
        )
    if len(entities) > 1 and hinted("relation"):
        relation_terms = _research_relation_terms(store, research, query, entities)
        return (
            EvidenceRequirement(
                "relation",
                description="; ".join(entities),
                retrieval_query=f"{' '.join(entities)} {_research_focus_query(store, research, query)}",
                required_entities=entities,
                # Entity co-occurrence is not a relationship.  The source row
                # must also carry every typed operation term selected by the
                # corpus policy for this relation family.
                required_operation_terms=relation_terms,
                authority_kinds=("normative_legal_text",),
                hierarchy_depth=3,
            ),
        )
    if len(segments) > 1 and not _single_support_covers_query(store, query):
        return tuple(
            EvidenceRequirement(
                f"dimension_{index}",
                description=segment,
                retrieval_query=_research_focus_query(store, research, segment),
                semantic_terms=_research_semantic_terms(store, research, segment, ()),
            )
            for index, segment in enumerate(segments, 1)
        )
    if hinted("procedure") and _procedure_applicable(research, normalized, entities):
        family = _procedure_family(research, normalized, entities)
        family_requirements = research.get("procedure_requirements_by_family", {}) if isinstance(research, dict) else {}
        stages = (
            family_requirements.get(family, ())
            if family and isinstance(family_requirements, dict)
            else research.get("procedure_requirements", ())
            if isinstance(research, dict)
            else ()
        )
        if isinstance(stages, (tuple, list)) and stages:
            return tuple(
                EvidenceRequirement(
                    str(stage["requirement_id"]),
                    description=str(stage.get("description") or stage["requirement_id"]),
                    retrieval_query=" ".join(
                        (*tuple(str(value) for value in stage.get("required_entities") or ()),
                         *tuple(str(value) for value in stage.get("support_terms") or ()))
                    ),
                    required_entities=tuple(str(value) for value in stage.get("required_entities") or ()),
                    support_terms=tuple(str(value) for value in stage.get("support_terms") or ()),
                    authority_kinds=("normative_legal_text",),
                    hierarchy_depth=int(stage["hierarchy_depth"]) if stage.get("hierarchy_depth") else None,
                )
                for stage in stages
                if isinstance(stage, dict) and stage.get("requirement_id")
            )
        try:
            minimum = max(1, int(minimums.get("procedure", 3)))
        except (TypeError, ValueError):
            minimum = 3
        return (
            EvidenceRequirement(
                "procedure",
                description="procedure",
                retrieval_query=_research_focus_query(store, research, query),
                required_entities=entities,
                semantic_terms=_research_semantic_terms(store, research, query, entities),
                support_terms=procedure_terms,
                authority_kinds=("normative_legal_text",),
                min_supports=minimum,
            ),
        )
    if entities and hinted("authority"):
        try:
            minimum = max(1, int(minimums.get("authority", 2)))
        except (TypeError, ValueError):
            minimum = 2
        entity = entities[0]
        return (
            EvidenceRequirement(
                "authority",
                description=entity,
                retrieval_query=_research_focus_query(store, research, query),
                required_entities=(entity,),
                contrast_entities=tuple(value for value in all_entities if value != entity),
                support_terms=authority_terms,
                entity_must_lead=True,
                authority_kinds=("normative_legal_text",),
                min_supports=minimum,
                allow_partial=True,
            ),
        )
    references = tuple(getattr(semantics, "legal_references", ()) or ())
    if len(references) > 1:
        return tuple(
            EvidenceRequirement(
                f"reference_{index}",
                description=reference,
                retrieval_query=reference,
                explicit_references=(reference,),
            )
            for index, reference in enumerate(references, 1)
        )
    return ()


def _procedure_applicable(research: dict, query: str, entities: tuple[str, ...]) -> bool:
    """Apply a configured procedure family only within its typed scope."""
    return _procedure_family(research, query, entities) is not None


def _procedure_family(research: dict, query: str, entities: tuple[str, ...]) -> str | None:
    """Return the corpus-owned procedure family matching the request."""
    families = research.get("procedure_applicability", {}) if isinstance(research, dict) else {}
    if not isinstance(families, dict):
        return None
    for name, family in families.items():
        if not isinstance(family, dict):
            continue
        required_entities = tuple(str(value) for value in family.get("required_entities") or ())
        signals = tuple(str(value) for value in family.get("signals") or ())
        if required_entities and not set(required_entities).intersection(entities):
            continue
        if signals and not contains_intent_phrase(query, signals):
            continue
        return str(name)
    return None


def _instrument_scope_roles(store: EvidenceStore, query: str) -> tuple[str, ...]:
    """Return every corpus-configured historical instrument named by a query."""
    intent = intent_config_for(getattr(store.config, "structured_strategy", "generic"), store.config)
    roles = []
    for role, pattern in intent.get("metadata_roles", ()):
        if str(role) in {"current_consolidated", "original_historical"}:
            continue
        if pattern.search(query or ""):
            roles.append(str(role))
    return tuple(dict.fromkeys(roles))


def _research_focus_query(store: EvidenceStore, research: dict, query: str) -> str:
    """Remove only corpus-configured task framing from a requirement query."""
    signals = research.get("semantic_hints", {}) if isinstance(research, dict) else {}
    excluded: set[str] = set()
    if isinstance(signals, dict):
        for name, values in signals.items():
            if name not in {"comparison", "procedure", "relation"}:
                continue
            if isinstance(values, (tuple, list)):
                for value in values:
                    if isinstance(value, str):
                        excluded.update(normalize_intent_text(value).split())
    summary = store.config.setting("document_summary", {}) or {}
    if isinstance(summary, dict):
        for value in summary.get("document_terms", ()) or ():
            if isinstance(value, str):
                excluded.update(normalize_intent_text(value).split())
    aliases = {
        normalize_intent_text(key): normalize_intent_text(value)
        for key, value in (store.config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    words = [word for word in meaningful_tokens(query, aliases=aliases) if word not in excluded]
    return " ".join(sorted(words)) or query


def _research_entity_labels(research: dict) -> tuple[str, ...]:
    aliases = research.get("entity_aliases", {}) if isinstance(research, dict) else {}
    return tuple(str(label) for label in aliases) if isinstance(aliases, dict) else ()


def _research_entities(research: dict, query: str) -> tuple[str, ...]:
    aliases = research.get("entity_aliases", {}) if isinstance(research, dict) else {}
    if not isinstance(aliases, dict):
        return ()
    found = []
    for label, values in aliases.items():
        terms = (str(label), *(str(value) for value in values if isinstance(value, str))) if isinstance(values, (tuple, list)) else (str(label),)
        if contains_intent_phrase(query, terms):
            found.append(str(label))
    return tuple(found)


def _research_semantic_terms(
    store: EvidenceStore,
    research: dict,
    query: str,
    entities: tuple[str, ...],
) -> tuple[str, ...]:
    aliases = {
        normalize_intent_text(key): normalize_intent_text(value)
        for key, value in (store.config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    excluded = {
        token
        for values in (research.get("semantic_hints", {}) or {}).values()
        if isinstance(values, (tuple, list))
        for value in values
        if isinstance(value, str)
        for token in meaningful_tokens(value, aliases=aliases)
    }
    excluded.update(
        token
        for entity in entities
        for token in meaningful_tokens(entity, aliases=aliases)
    )
    return tuple(sorted(meaningful_tokens(query, aliases=aliases) - excluded))


def _research_relation_terms(
    store: EvidenceStore,
    research: dict,
    query: str,
    entities: tuple[str, ...],
) -> tuple[str, ...]:
    """Keep operation vocabulary separate from ranking stopwords.

    A relation requirement needs one source-backed operation cue, while the
    ranker may still ignore common terms such as ``undang``.  Entity and
    relation framing are deliberately removed; the residual input is an
    immutable constraint rather than a second query or a synonym table.
    """
    aliases = {
        normalize_intent_text(key): normalize_intent_text(value)
        for key, value in (store.config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    ignored = {
        token
        for values in (research.get("semantic_hints", {}) or {}).values()
        if isinstance(values, (tuple, list))
        for value in values
        if isinstance(value, str)
        for token in tokens(value, aliases=aliases)
    }
    ignored.update(token for entity in entities for token in tokens(entity, aliases=aliases))
    operation_terms = research.get("relation_operation_terms", {})
    if isinstance(operation_terms, dict):
        for phrase, values in operation_terms.items():
            if not isinstance(phrase, str) or not isinstance(values, (tuple, list)):
                continue
            if contains_intent_phrase(query, (phrase,)):
                return tuple(
                    token
                    for value in values
                    if isinstance(value, str)
                    for token in tokens(value, aliases=aliases)
                    if token
                )
    return tuple(sorted({token for token in tokens(query, aliases=aliases) if len(token) > 2 and token not in ignored}))


def _semantic_support_excluded_terms(store: EvidenceStore, aliases: dict[str, str]) -> set[str]:
    policy = store.config.setting("lexical_normalization", {}) or {}
    return {
        token
        for phrase in policy.get("semantic_support_excluded_terms", ())
        if isinstance(phrase, str)
        for token in tokens(phrase, aliases=aliases)
    }


def _single_support_covers_query(store: EvidenceStore, query: str) -> bool:
    """Do not decompose a coordinated phrase already proved by one row."""
    aliases = {
        str(key).casefold(): str(value).casefold()
        for key, value in (store.config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    requested = meaningful_tokens(query, aliases=aliases)
    requested.difference_update(_semantic_support_excluded_terms(store, aliases))
    return any(
        validate_answer_candidate(store, row | {"route_sources": ("bm25",)})[0]
        and requested <= meaningful_tokens(
            " ".join(str(row.get(key) or "") for key in ("citation", "hierarchy", "quoted_text")),
            aliases=aliases,
        )
        for row in sparse_index_for_store(store).search(query, limit=10)
    )


def _apply_clarification_constraint(store: EvidenceStore, routed: dict, resolution: dict[str, str]) -> None:
    """Filter routed candidates without changing the user's semantic query."""
    legal_target = resolution.get("legal_target")
    concept_facet = resolution.get("concept_facet")
    if legal_target:
        units = {str(unit.get("legal_unit_id")): unit for unit in store.legal_units}

        def matches_target(row: dict) -> bool:
            unit = units.get(str(row.get("legal_unit_id") or ""))
            while unit is not None:
                if str(unit.get("unit_label") or "").casefold() == legal_target.casefold():
                    return True
                unit = units.get(str(unit.get("parent_legal_unit_id") or ""))
            return False

        routed["matches"] = tuple(row for row in routed.get("matches", ()) if matches_target(row))
    elif concept_facet:
        try:
            allowed = set(json.loads(concept_facet))
        except (TypeError, ValueError, json.JSONDecodeError):
            allowed = set()
        routed["matches"] = tuple(
            row
            for row in routed.get("matches", ())
            if str(row.get("evidence_id") or row.get("legal_unit_id")) in allowed
        )
    if resolution and not routed.get("matches"):
        routed["status"] = "no_results"


class LegalRuntimeService:
    def __init__(
        self,
        repo_root: Path | None = None,
        telemetry: Telemetry | None = None,
        strategy_registry: StrategyRegistry | None = None,
        *,
        answer_provider=None,
        external_wording: bool | None = None,
        planning_provider: ResearchPlanningProvider | None = None,
    ):
        self.registry = CorpusRegistry(repo_root, strategies=strategy_registry)
        self.repository = VerifiedCorpusRepository(self.registry)
        self.telemetry = telemetry or Telemetry.from_environment(self.registry)
        self.telemetry.bind_registry(self.registry)
        self._integrity_error: str | None = None
        self._store_cache: OrderedDict[str, EvidenceStore] = OrderedDict()
        self._store_cache_limit = max(1, len(self.registry.corpus_ids()))
        self._public_targets: OrderedDict[str, tuple[str, dict]] = OrderedDict()
        self._public_target_limit = 1024
        self._public_target_lock = RLock()
        self._catalog_service = None
        self._bookmarks = BookmarkRepository()
        self._external_wording = wording_enabled_from_environment() if external_wording is None else external_wording
        self._answer_provider = answer_provider
        self._planning_provider = planning_provider
        if self._external_wording and self._answer_provider is None:
            self._answer_provider = wording_provider_from_environment()

    def _store(self, corpus_id: str):
        cached = self._store_cache.get(corpus_id)
        if cached is not None:
            self._store_cache.move_to_end(corpus_id)
            self._integrity_error = None
            return cached
        try:
            config = self.repository.load(corpus_id).config
            self._integrity_error = None
        except CorpusIntegrityError as error:
            self._integrity_error = error.code
            self.telemetry.emit("integrity_failure", corpus_id=self._telemetry_corpus_id(corpus_id), reason_code=error.code)
            return None
        self.telemetry.emit("corpus_load", corpus_id=config.corpus_id, status="loaded")
        store = EvidenceStore.shared(config)
        self._store_cache[corpus_id] = store
        while len(self._store_cache) > self._store_cache_limit:
            self._store_cache.popitem(last=False)
        return store

    def _route_retrieval(self, corpus_id: str, query: str, store: EvidenceStore, **kwargs: Any) -> dict:
        result = route_retrieval(corpus_id, query, store, **kwargs)
        self.telemetry.emit("retrieval_route", corpus_id=self._telemetry_corpus_id(corpus_id), route=result["route"], status=result["status"])
        return result

    def research(
        self,
        corpus_id: str,
        query: str,
        *,
        intent: ResearchIntent | None = None,
        requirements: tuple[EvidenceRequirement, ...] = (),
        planning_provider: ResearchPlanningProvider | None = None,
        limit: int = 10,
        max_rounds: int | None = None,
    ) -> dict:
        """Run bounded retrieval variants and assess verified requirements."""
        store = self._store(corpus_id)
        if store is None:
            return {"status": "insufficient", "reason": self._integrity_error or "corpus_unavailable", "matches": (), "plan": None}
        def retrieve(variant_query, variant):
            route = variant.retrieval_lane
            variant_filters = {}
            if variant.source_role:
                variant_filters["source_role"] = variant.source_role
            if variant.temporal_scope:
                variant_filters["temporal_context"] = variant.temporal_scope
            result = self._route_retrieval(
                corpus_id,
                variant_query,
                store,
                limit=limit,
                route=route,
                metadata_filters=variant_filters or None,
                allow_structured_fallback=bool(variant.source_role),
            )
            if route == "dense" and result.get("status") == "dense_unavailable":
                fallback = self._route_retrieval(corpus_id, variant_query, store, limit=limit, route="auto")
                result = dict(fallback) | {"retrieval_degraded_reason": result.get("reason", "dense_unavailable")}
            if getattr(intent, "decomposition", False) and result.get("matches"):
                # A decomposition round may follow validated structural
                # relations from the lexical seed.  The seed remains a
                # candidate only; graph-expanded rows still pass the normal
                # support validator before assignment/publication.
                seeds = tuple(
                    dict(row) | {"route_sources": tuple(dict.fromkeys(("structured", *(row.get("route_sources") or ()))))}
                    for row in result.get("matches", ())
                    if row.get("evidence_id")
                )
                trace = graph_expand(store, seeds, {}, per_seed=max(1, limit), semantic=True)
                expanded = []
                for item in trace:
                    row = store.get(str(item.get("evidence_id") or ""))
                    if row is not None:
                        expanded.append(dict(row) | {"route_sources": ("graph",)})
                if expanded:
                    result = dict(result) | {"matches": tuple(result.get("matches", ())) + tuple(expanded)}
            return result

        result = execute_research_rounds(
            query,
            retrieve,
            store=store,
            intent=intent,
            provider=planning_provider,
            requirements=requirements,
            max_rounds=max_rounds,
        )
        assessment = result["sufficiency"]
        return {
            "status": assessment.status if assessment is not None else ("found" if result["matches"] else "insufficient"),
            "original_query": query,
            **result,
        }

    def _telemetry_corpus_id(self, corpus_id: str) -> str:
        config = self.registry.resolve(corpus_id)
        return config.corpus_id if config is not None else "unknown"

    def register_public_target(self, corpus_id: str, request: dict) -> str:
        """Return a stable opaque handle; persistence identifiers never leave this boundary."""
        encoded = json.dumps(
            {"corpus_id": corpus_id, "request": request},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        target = sha256(encoded).hexdigest()
        with self._public_target_lock:
            self._public_targets[target] = (corpus_id, dict(request))
            self._public_targets.move_to_end(target)
            while len(self._public_targets) > self._public_target_limit:
                self._public_targets.popitem(last=False)
        return target

    def public_identifier(self, corpus_id: str, kind: str, value: object) -> str:
        """Create a deterministic public identifier that has no storage meaning."""
        encoded = json.dumps(
            {"corpus_id": corpus_id, "kind": kind, "value": value},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _public_target_request(self, corpus_id: str, target: str | None) -> dict | None:
        if target is None:
            return None
        with self._public_target_lock:
            record = self._public_targets.get(target)
            if record is None or record[0] != corpus_id:
                return None
            self._public_targets.move_to_end(target)
            return dict(record[1])

    def public_clarification_context(self, corpus_id: str, target: str | None) -> dict | None:
        request = self._public_target_request(corpus_id, target)
        resolution = request.get("resolution") if request and request.get("kind") == "clarification_context" else None
        allowed = {"source_role", "legal_target", "relation_family", "entity", "temporal_scope", "concept_facet"}
        if not isinstance(resolution, dict) or set(resolution) - allowed:
            return None
        original_query = request.get("original_query") if request else None
        if not isinstance(original_query, str) or not original_query:
            return None
        return {
            "original_query": original_query,
            "resolution": {str(key): str(value) for key, value in resolution.items() if isinstance(value, str) and value},
        }

    def public_source_status_label(self, corpus_id: str, source_role: object) -> str | None:
        store = self._store(corpus_id)
        return _source_status_label({"source_role": source_role}, store) if store is not None else None

    def viewer_public(self, corpus_id: str, target: str | None) -> dict:
        if self._store(corpus_id) is None:
            return _integrity_failure(corpus_id, "", self._integrity_error)
        request = self._public_target_request(corpus_id, target)
        if request is None:
            return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
        result = self.viewer(corpus_id, **request)
        if result.get("pdf_access_available"):
            page_number = result.get("page_number") or (result.get("page_numbers") or (1,))[0]
            result["public_pdf_target"] = self.register_public_target(
                corpus_id,
                request | {"page_number": page_number},
            )
        return result

    def pdf_public(self, corpus_id: str, target: str | None) -> dict:
        if self._store(corpus_id) is None:
            return _integrity_failure(corpus_id, "", self._integrity_error)
        request = self._public_target_request(corpus_id, target)
        if request is None:
            return {"status": "not_found", "reason": "invalid_pdf_target", "corpus_id": corpus_id}
        return self.pdf_access(
            corpus_id,
            request.get("evidence_id"),
            relation_id=request.get("relation_id"),
            source_document_id=str(request.get("source_document_id") or ""),
            page_number=int(request.get("page_number") or 1),
            bbox_refs=tuple(request.get("bbox_refs") or ()),
        )

    def search(self, corpus_id: str, query: str, limit: int = 10, filters: dict | None = None) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, query, self._integrity_error) | {"results": ()}
        normalized_filters = normalize_filters(filters, config=store.config)
        if normalized_filters.get("_error"):
            return _catalog_search_response(
                corpus_id,
                query,
                (),
                "invalid_filter",
                normalized_filters["_error"],
                applied_filters=public_filters(normalized_filters),
                invalid_filters=normalized_filters.get("_invalid_filters", ()),
            )
        rows = _catalog_search(store, corpus_id, query, limit, normalized_filters)
        return _catalog_search_response(
            corpus_id,
            query,
            rows,
            "found" if rows else "no_results",
            None if rows else "document_not_found",
            applied_filters=public_filters(normalized_filters),
        )

    def catalog_search(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        *,
        corpus_id: str | None = None,
    ) -> dict:
        if corpus_id is not None and self._store(corpus_id) is None:
            return _integrity_failure(corpus_id, query, self._integrity_error)
        return self._catalog().search(query, limit, filters, corpus_id=corpus_id)

    def catalog_viewer(self, target: str):
        return self._catalog().document(target)

    def catalog_pdf(self, target: str) -> dict:
        return self._catalog().pdf(target)

    def catalog_documents(self):
        return self._catalog().repository.documents

    def catalog_document_for_source(self, source_role: object):
        role = str(source_role or "")
        return next(
            (
                document
                for document in self.catalog_documents()
                if document.identity.source_designation is not None
                and document.identity.source_designation.normalized_value == role
            ),
            None,
        )

    def catalog_document_for_target(self, corpus_id: str, target: str | None):
        catalog_document = self.catalog_viewer(str(target or ""))
        if catalog_document is not None:
            return catalog_document
        request = self._public_target_request(corpus_id, target)
        if request is None:
            return None
        store = self._store(corpus_id)
        if store is None:
            return None
        source_document_id = request.get("source_document_id")
        evidence = store.get(str(request.get("evidence_id") or ""))
        if not source_document_id and evidence:
            source_document_id = evidence.get("source_document_id")
        source = next(
            (row for row in store.source_documents if row.get("source_document_id") == source_document_id),
            None,
        )
        return self.catalog_document_for_source(source.get("source_role")) if source else None

    def citation_unit(self, corpus_id: str, row: dict):
        store = self._store(corpus_id)
        factory = getattr(getattr(store.config, "strategy", None), "citation_unit_factory", None) if store is not None else None
        return factory(store, row) if factory is not None else None

    def _catalog(self):
        if self._catalog_service is None:
            from tjipto.corpora.catalog import builtin_catalog

            self._catalog_service = CatalogService(builtin_catalog(self.registry.repo_root, self._store))
        return self._catalog_service

    def citation(
        self,
        corpus_id: str,
        query: str,
        source_role: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, query, self._integrity_error) | _empty_citation_fields()
        scope = scope_guard_context(store, query)
        if scope:
            return {
                "status": "citation_not_found",
                "public_status": "insufficient_evidence",
                "route": scope["route"],
                "intent": scope["intent"],
                "matches": (),
                "reason": scope["reason"],
                "requested_function": scope.get("requested_function"),
                "target_reference": scope.get("target_reference"),
                "legal_domain": scope.get("legal_domain"),
                **_empty_citation_fields(),
            }
        metadata_filters = dict(filters or {})
        scope = resolve_source_scope(query, strategy=getattr(store.config, "query_strategy", "generic"), config=store.config)
        requested_role = source_role or (scope.role if scope.explicit else None)
        if requested_role is not None:
            metadata_filters["source_role"] = requested_role
        routed = self._route_retrieval(corpus_id, query, store, metadata_filters=metadata_filters)
        if routed["intent"] != "exact_citation":
            return routed | {
                "status": "citation_not_found",
                "route": "citation_not_found",
                "matches": (),
                "reason": "not_a_citation",
                **_empty_citation_fields(),
            }
        if not routed["matches"]:
            return routed | {"status": routed["status"], **_empty_citation_fields()}
        context_pack = assemble_context_pack(store, routed["matches"])
        citations = tuple(_citation_with_authority(store, row) for row in context_pack["citation_payloads"])
        return routed | {
            "status": "found",
            "context_pack": context_pack,
            "citation_payloads": citations,
            "viewer_refs": context_pack["viewer_refs"],
            "validation_reasons": context_pack["validation_reasons"],
        }

    def viewer(
        self,
        corpus_id: str,
        evidence_id: str | None = None,
        *,
        support_unit_id: str | None = None,
        source_support_id: str | None = None,
        relation_id: str | None = None,
        source_document_id: str | None = None,
        page_number: int | None = None,
        bbox_id: str | None = None,
        bbox_refs: tuple[str, ...] = (),
        proposition_id: str | None = None,
        quoted_text: str | None = None,
        support_projection: dict | None = None,
        source_pdf_path: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error)
        support = store.meaningful_support_unit(support_unit_id) if support_unit_id else None
        if support_unit_id and (
            support is None
            or support.get("decision_kind") == "typed_exclusion"
            or support.get("viewer_eligible") is not True
        ):
            return {"status": "not_found", "reason": "invalid_support_target", "corpus_id": corpus_id}
        if support is not None:
            source_document_id = str(support["source_document_id"])
            page_number = int(support["page_numbers"][0])
            bbox_refs = tuple(support.get("bbox_refs") or ())
            quoted_text = "\n".join(
                str(store.page_text_span(span_id).get("exact_quote") or "")
                for span_id in support.get("text_span_ids") or ()
                if store.page_text_span(span_id) is not None
            )
            if support.get("owner_type") == "review_decision":
                raw = store.source_span(str((support.get("raw_source_span_ids") or ("",))[0]))
                evidence_id = str(raw.get("source_support_id")) if raw else None
            else:
                evidence_id = str(support["owner_id"])
            if support.get("bbox_precision") == "page_grounded_only":
                source = _source_document_by_id(store, source_document_id)
                if source is None:
                    return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
                return document_viewer_payload(store, corpus_id, source, page_number=page_number) | {
                    "quoted_text": quoted_text,
                    "page_numbers": tuple(support["page_numbers"]),
                    "source_role": support["source_role"],
                    "temporal_context": support["temporal_context"],
                }
        evidence_id = evidence_id or source_support_id
        if evidence_id is None:
            source = _source_document_by_id(store, source_document_id)
            if source is None:
                return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
            return document_viewer_payload(
                store,
                corpus_id,
                source,
                page_number=page_number,
                source_pdf_path=source_pdf_path,
            )
        evidence = store.get(evidence_id)
        if evidence is None:
            evidence = _metadata_grounding_evidence(store, evidence_id)
        synthetic_bboxes: list[dict] | None = list(support.get("bbox_rectangles") or ()) if support is not None else None
        if evidence is None:
            evidence, synthetic_bboxes = _source_conflict_viewer_evidence(store, evidence_id)
        if evidence is None:
            evidence = _source_span_evidence(store, evidence_id)
            synthetic_bboxes = store.source_span_bboxes(evidence_id) if evidence is not None else None
        if evidence is None:
            return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
        if proposition_id is not None:
            proposition = next(
                (
                    row
                    for row in store.propositions
                    if row.get("proposition_id") == proposition_id
                    and row.get("legal_unit_id") == evidence.get("legal_unit_id")
                    and row.get("source_document_id") == evidence.get("source_document_id")
                    and tuple(row.get("bbox_refs") or ()) == bbox_refs
                ),
                None,
            )
            overlay = viewer_overlay_rectangles(proposition or {})
            if proposition is None or not overlay:
                return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
            evidence = evidence | {
                "bbox_refs": bbox_refs,
                "quoted_text": proposition.get("exact_quote"),
                "page_numbers": tuple(proposition.get("page_numbers") or ()),
            }
            synthetic_bboxes = list(overlay)
        relation = _relation_for_evidence(store, evidence_id, relation_id)
        if relation_id is not None and relation is None:
            return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
        if relation is not None:
            evidence = evidence | {
                "bbox_refs": tuple(relation.get("bbox_refs") or ()),
                "quoted_text": relation.get("quoted_text") or evidence.get("quoted_text"),
            }
        if quoted_text is not None:
            evidence = evidence | {"quoted_text": quoted_text}
        if support is not None:
            evidence = evidence | {
                "bbox_refs": bbox_refs,
                "bbox_precision": support["bbox_precision"],
                "viewer_highlightable": support.get("highlight_eligible") is True,
                "page_numbers": tuple(support["page_numbers"]),
                "source_document_id": support["source_document_id"],
                "source_role": support["source_role"],
                "temporal_context": support["temporal_context"],
            }
            synthetic_bboxes = list(support.get("bbox_rectangles") or ())
        if support_projection:
            evidence = evidence | {
                key: support_projection[key]
                for key in (
                    "display_text", "copy_text", "layout_lines", "presentation_as_legal_quote",
                    "citation_final", "relevant_quote_eligible",
                )
                if key in support_projection
            }
        bboxes = (
            synthetic_bboxes
            if synthetic_bboxes is not None
            else store.metadata_bboxes_for(evidence_id)
            if evidence.get("metadata_grounding")
            else store.bboxes_for_refs(tuple(relation.get("bbox_refs") or ())) if relation is not None else store.bboxes_for(evidence_id)
        )
        if bbox_refs and proposition_id is None and support is None:
            bboxes = store.bboxes_for_refs(bbox_refs)
        if proposition_id is None:
            bboxes = _select_viewer_bboxes(bboxes, bbox_refs)
        if bbox_refs and not bboxes:
            return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
        return _viewer_with_authority(
            store,
            evidence,
            viewer_payload(
                store,
                corpus_id,
                evidence,
                bboxes,
                source_document_id=source_document_id,
                page_number=page_number,
                bbox_id=bbox_id,
                source_pdf_path=source_pdf_path,
            ),
        )

    def pdf_access(
        self,
        corpus_id: str,
        evidence_id: str | None,
        *,
        source_support_id: str | None = None,
        relation_id: str | None = None,
        source_document_id: str,
        page_number: int,
        source_sha256: str | None = None,
        bbox_id: str | None = None,
        bbox_refs: tuple[str, ...] = (),
        source_pdf_path: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error)
        evidence_id = evidence_id or source_support_id
        if evidence_id is None:
            source = _source_document_by_id(store, source_document_id)
            if source is None:
                return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
            return resolve_document_pdf_access(
                store,
                corpus_id,
                source,
                page_number=page_number,
                source_pdf_path=source_pdf_path,
            )
        evidence = store.get(evidence_id)
        if evidence is None:
            evidence = _metadata_grounding_evidence(store, evidence_id)
        synthetic_bboxes: list[dict] | None = None
        if evidence is None:
            evidence, synthetic_bboxes = _source_conflict_viewer_evidence(store, evidence_id)
        if evidence is None:
            evidence = _source_span_evidence(store, evidence_id)
            synthetic_bboxes = store.source_span_bboxes(evidence_id) if evidence is not None else None
        if evidence is None:
            return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
        relation = _relation_for_evidence(store, evidence_id, relation_id)
        if relation_id is not None and relation is None:
            return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
        if relation is not None:
            evidence = evidence | {
                "bbox_refs": tuple(relation.get("bbox_refs") or ()),
                "quoted_text": relation.get("quoted_text") or evidence.get("quoted_text"),
            }
        bboxes = (
            synthetic_bboxes
            if synthetic_bboxes is not None
            else store.metadata_bboxes_for(evidence_id)
            if evidence.get("metadata_grounding")
            else store.bboxes_for_refs(tuple(relation.get("bbox_refs") or ())) if relation is not None else store.bboxes_for(evidence_id)
        )
        if bbox_refs:
            bboxes = store.bboxes_for_refs(bbox_refs)
        bboxes = _select_viewer_bboxes(bboxes, bbox_refs)
        if bbox_refs and not bboxes:
            return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
        return resolve_pdf_access(
            store,
            corpus_id,
            evidence,
            bboxes,
            source_document_id=source_document_id,
            page_number=page_number,
            bbox_id=bbox_id,
            source_sha256=source_sha256,
            source_pdf_path=source_pdf_path,
        )

    def capabilities(self, corpus_id: str) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error) | {"capabilities": ()}
        return {
            "status": "ok",
            "corpus_id": corpus_id,
            "readiness": True,
            "manifest_digest": store.config.manifest_digest,
            "artifact_set_digest": store.config.artifact_set_digest,
            "artifact_access_mode": store.config.artifact_access_mode,
            "canonical_build_eligible": store.config.canonical_build_eligible,
            "capabilities": ("search", "ask", "citation", "viewer", "bookmarks"),
        }

    def bookmarks(self, corpus_id: str) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error) | {"bookmarks": ()}
        snapshot = self._bookmarks.list(corpus_id)
        bookmarks = tuple(sorted((self._bookmark_status(row, store) for row in snapshot), key=lambda row: row["bookmark_id"]))
        return {
            "status": "ok",
            "corpus_id": corpus_id,
            "persistence": "memory",
            "persistence_label": "temporary_process_memory",
            "bookmarks": bookmarks,
        }

    def bookmark(
        self,
        corpus_id: str,
        evidence_id: str,
        note: str | None = None,
        citation_id: str | None = None,
        viewer_ref_id: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error)
        evidence = store.get(evidence_id)
        if evidence is None or evidence.get("status") != "final":
            return {"status": "unavailable", "reason": "evidence_unavailable", "corpus_id": corpus_id}
        bookmark = {
            "bookmark_id": f"bm_{uuid4().hex}",
            "corpus_id": corpus_id,
            "legal_unit_id": evidence.get("legal_unit_id"),
            "evidence_id": evidence_id,
            "citation_id": citation_id or evidence_id,
            "viewer_ref_id": viewer_ref_id or evidence_id,
            "note": note,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "active",
        }
        self._bookmarks.save(bookmark)
        return {"status": "saved", "bookmark": bookmark}

    def bookmark_public(self, corpus_id: str, target: str | None, note: str | None = None) -> dict:
        request = self._public_target_request(corpus_id, target)
        if request is not None and request.get("evidence_id"):
            result = self.bookmark(corpus_id, request["evidence_id"], note)
            if result.get("status") != "saved":
                return result
            result["bookmark"]["public_target"] = target
            return result
        if target is None or self.catalog_viewer(target) is None:
            return {"status": "unavailable", "reason": "bookmark_target_unavailable"}
        bookmark = {
            "bookmark_id": f"bm_{uuid4().hex}",
            "corpus_id": corpus_id,
            "target_kind": "catalog_document",
            "public_target": target,
            "note": note,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "active",
        }
        self._bookmarks.save(bookmark)
        return {"status": "saved", "bookmark": bookmark}

    def delete_bookmark_public(self, corpus_id: str, public_bookmark_id: str | None) -> dict:
        if not public_bookmark_id:
            return {"status": "unavailable"}
        deleted = self._bookmarks.delete_public(
            corpus_id,
            public_bookmark_id,
            lambda bookmark_id: self.public_identifier(corpus_id, "bookmark", bookmark_id),
        )
        if not deleted:
            return {"status": "unavailable"}
        return {"status": "deleted", "public_bookmark_id": public_bookmark_id}

    def _bookmark_status(self, bookmark: dict, store=None) -> dict:
        if bookmark.get("target_kind") == "catalog_document":
            status = "active" if self.catalog_viewer(str(bookmark.get("public_target") or "")) is not None else "unavailable"
            return bookmark | {"status": status}
        store = store or self._store(bookmark["corpus_id"])
        evidence = store.get(bookmark["evidence_id"]) if store else None
        status = "active" if evidence and evidence.get("status") == "final" else "unavailable"
        return bookmark | {"status": status}

    def ask(
        self,
        corpus_id: str,
        query: str,
        limit: int = 3,
        filters: dict | None = None,
        clarification: dict[str, str] | None = None,
        evidence_requirements: tuple[EvidenceRequirement, ...] = (),
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, query, self._integrity_error)
        source_text = source_text_response(store, corpus_id, query)
        if source_text is not None:
            return source_text
        normalized_summary = document_summary_query(
            query,
            strategy=getattr(store.config, "query_strategy", "generic"),
            config=store.config,
        )
        if normalized_summary and normalized_summary != query:
            result = self.ask(
                corpus_id,
                normalized_summary,
                limit,
                filters,
                clarification,
                evidence_requirements,
            )
            return result | {"original_query": query, "normalized_query": normalized_summary}
        semantics = interpret_query(store, corpus_id, query, available_corpora=self.registry.corpus_ids())
        anomaly = _source_anomaly_response(store, corpus_id, query)
        if anomaly:
            return anomaly
        source_document = source_document_response(
            store,
            corpus_id,
            query,
            has_resolved_target=_has_resolved_legal_target(corpus_id, query, config=store.config),
            document_title=_document_title,
            insufficient_answer=_answer_templates(store)["insufficient"],
        )
        if source_document:
            return source_document
        # A resolved legal target has precedence over the instrument classifier.
        # Amendment wording then scopes the structured lookup to that source role.
        resolution = clarification or {}
        relation_family = resolution.get("relation_family")
        amendment_target = amendment_relation_target(store, query, relation_family=relation_family)
        instrument = None if (
            semantics.requested_function == "temporal_quotation"
            or _has_resolved_legal_target(corpus_id, query, config=store.config)
            or amendment_target.get("mode") is not None
            or len(_instrument_scope_roles(store, query)) > 1
        ) else _instrument_intent_context(store, query)
        if instrument:
            row, route, reason = instrument
            templates = _answer_templates(store)
            if row is None:
                context_pack = empty_context_pack(reason)
                return {
                    "status": "insufficient_evidence",
                    "route": route,
                    "intent": "instrument_unit_lookup",
                    "corpus_id": corpus_id,
                    "original_query": query,
                    "normalized_query": query.strip(),
                    "matches": (),
                    "reason": reason,
                    "answer_type": "none",
                    "answer": templates["insufficient"],
                    "context_pack": context_pack,
                    "evidence": (),
                    "citations": (),
                    "final_citations": (),
                    "historical_citations": (),
                    "metadata_support": (),
                    "structural_support": (),
                    "trace_support": (),
                    "viewer_refs": (),
                    "metadata_facts": (),
                    "legal_relations": (),
                    "answer_scope": "insufficient_evidence",
                    "warnings": (),
                    "insufficient_reasons": (reason,),
                }
            context_pack = assemble_context_pack(store, (row,))
            evidence = context_pack["answer_evidence"]
            if not evidence:
                return {
                    "status": "insufficient_evidence",
                    "route": route,
                    "intent": "instrument_unit_lookup",
                    "corpus_id": corpus_id,
                    "original_query": query,
                    "normalized_query": query.strip(),
                    "matches": (row,),
                    "reason": reason,
                    "answer_type": "none",
                    "answer": templates["insufficient"],
                    "context_pack": context_pack,
                    "evidence": (),
                    "citations": (),
                    "final_citations": (),
                    "historical_citations": context_pack.get("historical_citations", ()),
                    "metadata_support": context_pack.get("metadata_support", ()),
                    "structural_support": context_pack.get("structural_support", ()),
                    "trace_support": context_pack.get("trace_support", ()),
                    "viewer_refs": (),
                    "metadata_facts": (),
                    "legal_relations": (),
                    "answer_scope": "insufficient_evidence",
                    "warnings": (),
                    "insufficient_reasons": (reason,),
                }
            instrument_status = "answer_ready" if context_pack["citation_payloads"] else "limited_answer"
            return {
                "status": instrument_status,
                "route": route,
                "intent": "instrument_unit_lookup",
                "corpus_id": corpus_id,
                "original_query": query,
                "normalized_query": query.strip(),
                "matches": (row,),
                "reason": None,
                "answer_type": "quoted_evidence",
                "answer": self._answer_text(store, instrument_status, evidence, templates),
                "context_pack": context_pack,
                "evidence": evidence,
                "citations": tuple(_citation_with_authority(store, item) for item in context_pack["citation_payloads"]),
                "final_citations": tuple(_citation_with_authority(store, item) for item in context_pack["citation_payloads"]),
                "historical_citations": context_pack.get("historical_citations", ()),
                "metadata_support": context_pack.get("metadata_support", ()),
                "structural_support": context_pack.get("structural_support", ()),
                "trace_support": context_pack.get("trace_support", ()),
                "viewer_refs": context_pack["viewer_refs"] if context_pack["citation_payloads"] else (),
                "metadata_facts": (),
                "legal_relations": (),
                "answer_scope": "direct_evidence" if instrument_status == "answer_ready" else "limited_evidence",
                "warnings": (),
                "insufficient_reasons": (),
            }
        active_requirements = tuple(evidence_requirements)
        if not active_requirements:
            active_requirements = _research_requirements_for_ask(store, semantics, query)
        semantic_scope_loss = not _semantic_scope_covered(store, semantics, query, active_requirements)
        research_intent = _research_intent_for_ask(store, semantics, query, active_requirements)
        research_routed = None
        if research_intent.complex or active_requirements:
            research_result = self.research(
                corpus_id,
                query,
                intent=research_intent,
                requirements=active_requirements,
                planning_provider=self._planning_provider,
                limit=_research_candidate_limit(store, query, _clarification_candidate_limit(store, query, limit)),
            )
            if research_result.get("routes"):
                research_routed = dict(research_result["routes"][0])
                research_routed["matches"] = research_result.get("matches", ())
                research_routed["status"] = "found" if research_routed["matches"] else "no_results"
                research_routed["research_plan"] = research_result.get("plan")
                research_routed["research_stop_reason"] = research_result.get("stop_reason")
                research_routed["semantic_scope_loss"] = semantic_scope_loss
        scope = scope_guard_context(store, query, capability=semantics.capability_decision)
        scoped_routed = None
        if scope:
            # Scope is a conclusion from the retrieved candidates, never a
            # placeholder retrieval attempt.
            scoped_routed = self._route_retrieval(
                corpus_id,
                query,
                store,
                limit=_clarification_candidate_limit(store, query, limit),
                metadata_filters=filters,
                allow_navigation=semantics.requested_function != "temporal_quotation",
                allow_relation=semantics.requested_function != "temporal_quotation",
                relation_family=relation_family,
            )
            scoped_routed["original_query"] = query
            _apply_clarification_constraint(store, scoped_routed, resolution)
            if (
                scope["route"] == "current_fact_unsupported"
                or semantics.capability_decision.missing_capabilities
                or not _scope_has_verified_support(store, scoped_routed)
            ):
                decision = (
                    clarification_decision(store, semantics, scoped_routed)
                    if scope["route"] != "current_fact_unsupported" and not semantics.capability_decision.missing_capabilities
                    else None
                )
                if decision:
                    return scoped_routed | _clarification_response(scoped_routed, decision)
                templates = _answer_templates(store)
                capability = semantics.capability_decision
                missing_corpora = capability.missing_corpora
                missing_capabilities = capability.missing_capabilities
                missing_domain = bool(missing_corpora)
                reason = "missing_corpus_support" if missing_domain else scope["reason"]
                context_pack = empty_context_pack(reason)
                return scope | {
                    "status": "insufficient_evidence",
                    "route": "missing_corpus" if missing_domain else scope["route"],
                    "reason": reason,
                    "reason_code": reason,
                    "corpus_id": corpus_id,
                    "original_query": query,
                    "normalized_query": query.strip(),
                    "matches": (),
                    "answer_type": "none",
                    "answer": templates["insufficient"],
                    "context_pack": context_pack,
                    "evidence": (),
                    "citations": (),
                    "final_citations": (),
                    "historical_citations": context_pack.get("historical_citations", ()),
                    "metadata_support": context_pack.get("metadata_support", ()),
                    "structural_support": context_pack.get("structural_support", ()),
                    "trace_support": context_pack.get("trace_support", ()),
                    "viewer_refs": (),
                    "metadata_facts": (),
                    "legal_relations": (),
                    "answer_scope": "insufficient_evidence",
                    "warnings": (),
                    "insufficient_reasons": (reason,),
                    "capability_decision": capability.public(),
                    "available_corpora": semantics.available_corpora,
                    "needed_corpora": missing_corpora,
                    "missing_corpora": missing_corpora,
                    "required_capabilities": capability.required_capabilities,
                    "missing_capabilities": missing_capabilities,
                    "retrieval_attempted": True,
                    "retrieval_route": scoped_routed["route"],
                    "retrieval_candidate_count": len(scoped_routed["matches"]),
                }
        semantic_filters = dict(filters or {})
        if resolution.get("source_role"):
            semantic_filters["source_role"] = resolution["source_role"]
        if semantics.source_role and "source_role" not in semantic_filters:
            semantic_filters["source_role"] = semantics.source_role
        routed = scoped_routed or research_routed or self._route_retrieval(
            corpus_id,
            query,
            store,
            limit=_clarification_candidate_limit(store, query, limit),
            metadata_filters=semantic_filters,
            allow_navigation=semantics.requested_function != "temporal_quotation",
            allow_relation=semantics.requested_function != "temporal_quotation",
            relation_family=relation_family,
        )
        _apply_clarification_constraint(store, routed, resolution)
        if resolution.get("entity"):
            routed["matches"] = tuple(
                row for row in routed.get("matches", ()) if row.get("entity_identity") == resolution["entity"]
            )
        if resolution.get("temporal_scope"):
            routed["matches"] = tuple(
                row for row in routed.get("matches", ()) if row.get("temporal_context") == resolution["temporal_scope"]
            )
        if not routed.get("matches") and resolution:
            routed["status"] = "no_results"
        if semantic_scope_loss:
            routed["semantic_scope_loss"] = True
        evidence_set = collect_evidence_set(store, routed.get("matches", ()), active_requirements) if active_requirements else None
        assessment = assess_sufficiency(evidence_set, active_requirements) if evidence_set is not None else None
        if evidence_set is not None and assessment is not None:
            routed["evidence_set"] = {
                "support_ids": tuple(str(row.get("evidence_id")) for row in evidence_set.supports),
                "assignments": evidence_set.assignments,
                "missing_requirement_ids": evidence_set.missing_requirement_ids,
                "missing_reasons": evidence_set.missing_reasons,
            }
            routed["sufficiency"] = {
                "status": assessment.status,
                "fulfilled_requirement_ids": assessment.fulfilled_requirement_ids,
                "missing_requirement_ids": assessment.missing_requirement_ids,
                "missing_reasons": assessment.missing_reasons,
                "retry_allowed": assessment.retry_allowed,
            }
        routed["matches"] = tuple(
            {key: value for key, value in row.items() if not str(key).startswith("_")}
            for row in routed.get("matches", ())
        )
        ask_route = _ask_route(routed["route"])
        templates = _answer_templates(store)
        routed["original_query"] = query
        decision = None if clarification else clarification_decision(store, semantics, routed)
        if decision:
            return routed | _clarification_response(routed, decision)
        if routed.get("route") == "document_relation":
            return _relation_response(store, routed)
        if routed["status"] != "found":
            public_status = (
                "insufficient_evidence"
                if routed.get("route")
                in {"metadata_not_found", "relation_not_found", "structured_not_found", "scope_unresolved"}
                else routed["status"]
            )
            context_pack = empty_context_pack(routed.get("reason") or routed["status"])
            return project_response(
                routed,
                AnswerDecision(
                    public_status,
                    ask_route,
                    "none",
                    templates["insufficient"],
                    context_pack,
                    insufficient_reasons=(assessment.missing_requirement_ids if assessment is not None and assessment.missing_requirement_ids else (routed.get("reason") or routed["status"],)),
                ),
            )
        if routed.get("semantic_scope_loss"):
            context_pack = empty_context_pack("semantic_scope_loss")
            return project_response(
                routed,
                AnswerDecision(
                    "insufficient_evidence",
                    ask_route,
                    "none",
                    templates["insufficient"],
                    context_pack,
                    insufficient_reasons=("semantic_scope_loss",),
                ),
            )
        if evidence_set is not None and assessment is not None and assessment.status == "insufficient":
            context_pack = empty_context_pack("required_evidence_missing")
            return project_response(
                routed,
                AnswerDecision(
                    "insufficient_evidence",
                    ask_route,
                    "none",
                    templates["insufficient"],
                    context_pack,
                    insufficient_reasons=tuple(assessment.missing_requirement_ids),
                ),
            )
        answer_matches = evidence_set.supports if evidence_set is not None else routed["matches"]
        if ask_route == "lexical_fallback" and evidence_set is None:
            # BM25 may rank several independently relevant rows, but one
            # answer may claim only one complete source-backed proposition.
            candidates = tuple(
                row
                for row in answer_matches
                if validate_answer_candidate(store, row)[0]
                and (
                    semantics.requested_function == "proposition_verification"
                    or _semantic_supports_query(store, query, row)
                )
            )
            answer_matches = min(candidates, key=_semantic_support_rank) if candidates else None
            answer_matches = (answer_matches,) if answer_matches else ()
        context_pack = assemble_context_pack(store, answer_matches)
        evidence = context_pack["answer_evidence"]
        if not evidence:
            reasons = tuple(sorted(set(context_pack["validation_reasons"].values()))) or ("semantic_support_missing",)
            return project_response(
                routed,
                AnswerDecision(
                    "insufficient_evidence",
                    ask_route,
                    "none",
                    templates["insufficient"],
                    context_pack,
                    insufficient_reasons=reasons,
                ),
            )
        claim_support = verify_claims(semantics, evidence, store)
        if not all_supported(claim_support):
            claim_reason = next(claim.reason_code for claim in claim_support if claim.reason_code)
            return project_response(
                routed,
                AnswerDecision(
                    "insufficient_evidence",
                    ask_route,
                    "none",
                    _claim_answer(claim_support),
                    empty_context_pack(claim_reason),
                    insufficient_reasons=(claim_reason,),
                    reason_code=claim_reason,
                    claim_support=tuple(claim.public() for claim in claim_support),
                ),
            )
        status = (
            "limited_answer"
            if assessment is not None
            and assessment.status == "complete"
            and any(requirement.allow_partial for requirement in active_requirements)
            else "answer_ready"
            if assessment is not None and assessment.status == "complete"
            else "limited_answer"
            if ask_route == "lexical_fallback" or (context_pack["trace_support"] and not context_pack["citation_payloads"])
            else "answer_ready"
        )
        metadata_support = tuple(_metadata_support(store, row) for row in evidence if row.get("metadata_field"))
        citations: tuple[dict, ...]
        if metadata_support:
            citations = ()
            viewer_refs = ()
        else:
            citations = _claim_citations(
                tuple(_citation_with_authority(store, row) for row in context_pack["citation_payloads"]),
                claim_support,
            )
            viewer_refs = tuple(row["viewer_ref"] for row in citations)
        if metadata_support:
            deterministic_answer = _metadata_answer(store, metadata_support)
        elif evidence_set is not None:
            deterministic_answer = _research_answer(
                store,
                evidence,
                evidence_set,
                active_requirements,
                assessment,
            )
        else:
            deterministic_answer = self._answer_text(store, status, evidence, templates, claim_support)
            if routed.get("route") == "structure_list":
                labels = tuple(dict.fromkeys(
                    str(row.get("citation") or row.get("label") or "").strip() for row in evidence
                ))
                labels = tuple(label for label in labels if label)
                if labels:
                    deterministic_answer = ", ".join(labels)
        answer = self._agent_answer(evidence, deterministic_answer)
        return project_response(
            routed,
            AnswerDecision(
                status,
                ask_route,
                _answer_type(ask_route, status),
                answer,
                context_pack,
                evidence=evidence,
                citations=citations,
                final_citations=citations,
                historical_citations=context_pack.get("historical_citations", ()),
                viewer_refs=viewer_refs,
                metadata_facts=tuple(_metadata_fact(row) for row in evidence if row.get("metadata_field")),
                metadata_support=metadata_support,
                structural_support=tuple(_citation_with_authority(store, row) for row in context_pack.get("structural_support", ())),
                trace_support=tuple(_citation_with_authority(store, row) for row in context_pack.get("trace_support", ())),
                legal_relations=tuple(row["legal_relation"] for row in evidence if row.get("legal_relation")),
                answer_scope="direct_evidence" if status == "answer_ready" else "limited_evidence",
                warnings=("metadata_support_not_exact_highlightable",)
                if any(row.get("viewer_highlightable") is not True for row in metadata_support)
                else (),
                claim_support=tuple(claim.public() for claim in claim_support),
            ),
        )

    def _agent_answer(self, evidence: tuple[dict, ...], fallback: str) -> str:
        if not self._external_wording or self._answer_provider is None:
            return fallback
        facts = _verified_answer_facts(evidence, fallback)
        try:
            proposal = self._answer_provider.propose(json.dumps({"facts": facts}, ensure_ascii=False, sort_keys=True))
        except Exception:
            return fallback
        return _render_wording(proposal, fallback, facts)

    def _answer_text(
        self,
        store,
        status: str,
        evidence: tuple[dict, ...],
        templates: dict[str, str],
        claims=(),
    ) -> str:
        if evidence[0].get("metadata_answer"):
            return _metadata_answer(store, tuple(evidence))
        if evidence[0].get("legal_relation"):
            relations = tuple(row["legal_relation"] for row in evidence)
            sources = tuple(dict.fromkeys(str(row.get("source_label") or "") for row in relations if row.get("source_label")))
            targets = tuple(dict.fromkeys(str(row.get("target_label") or "") for row in relations if row.get("target_label")))
            return f"{sources[0]} memuat: {', '.join(targets)}." if len(sources) == 1 and targets else templates["legal_relation"]
        quote = " ".join(_compact_text(row.get("quoted_text") or row.get("display_text") or "") for row in evidence).strip()
        if claims:
            claim = claims[0]
            segment = next((item.get("exact_quote") for item in claim.support_segments if item.get("exact_quote")), None)
            return f"Klaim “{claim.claim_text}” didukung oleh segmen terverifikasi: {segment or quote}."
        source_label = _source_status_label(evidence[0], store)
        citation = evidence[0].get("label") or evidence[0].get("citation") or "Bukti"
        prefix = f"{source_label} — " if evidence[0].get("source_role") != "current_consolidated" and source_label else ""
        return f"{prefix}{citation}: {quote}" if quote else templates["citation"].format(citation=citation)


def _compact_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _semantic_supports_query(store: EvidenceStore, query: str, row: dict) -> bool:
    """Guard the single-row lexical fallback after safe lexical normalization."""
    aliases = {
        str(key).casefold(): str(value).casefold()
        for key, value in (store.config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    requested = meaningful_tokens(query, aliases=aliases)
    requested.difference_update(_semantic_support_excluded_terms(store, aliases))
    source = " ".join(
        str(value or "")
        for value in (row.get("citation"), " ".join(row.get("hierarchy") or ()), row.get("quoted_text"))
    )
    supported = meaningful_tokens(source, aliases=aliases)
    if "boleh" in requested and re.search(r"\bboleh(?:kah)?\b", normalize_intent_text(query)):
        requested.discard("boleh")
    related = (store.config.setting("lexical_normalization", {}) or {}).get("related_terms", {})
    for term in tuple(requested):
        alternatives = {
            token
            for value in related.get(term, ())
            if isinstance(value, str)
            for token in meaningful_tokens(value, aliases=aliases)
        }
        if alternatives & supported:
            requested.discard(term)
    # Typed EvidenceRequirement assignment owns complex-answer completeness.
    # This guard is only for a one-row lexical fallback, where every retained
    # content token must be grounded after safe aliases and question auxiliaries
    # have been removed by the lexical policy.
    return bool(requested and requested <= supported)


def _semantic_support_rank(row: dict) -> tuple[int, int, str]:
    return (
        1 if row.get("authority_kind") == "structural_context" else 0,
        len(_compact_text(row.get("quoted_text"))),
        str(row.get("evidence_id") or ""),
    )


def _metadata_answer(store, rows: tuple[dict, ...]) -> str:
    names = _unique_printed_names(rows)
    values = names or tuple(
        dict.fromkeys(
            _compact_text(row.get("display_text") or row.get("answer") or row.get("metadata_answer"))
            for row in rows
            if row.get("display_text") or row.get("answer") or row.get("metadata_answer")
        )
    )
    source_label = _source_status_label(rows[0], store) if rows else None
    suffix = f" Sumber: {source_label}." if source_label else ""
    return f"{', '.join(values)}.{suffix}" if values else "Bukti metadata terverifikasi tidak memuat nilai."


def _research_answer(
    store,
    evidence: tuple[dict, ...],
    evidence_set,
    requirements: tuple[EvidenceRequirement, ...],
    assessment,
) -> str:
    """Compose only requirement-assigned, source-backed findings."""
    by_id = {str(row.get("evidence_id")): row for row in evidence}
    requirement_by_id = {row.requirement_id: row for row in requirements}
    findings = []
    labels = []
    for requirement_id, support_ids in evidence_set.assignments:
        requirement = requirement_by_id.get(requirement_id)
        heading = requirement.description if requirement and requirement.description else requirement_id.replace("_", " ")
        rows = tuple(by_id[support_id] for support_id in support_ids if support_id in by_id)
        for row in rows:
            label = str(row.get("label") or row.get("citation") or "Ketentuan")
            quote = _compact_text(row.get("quoted_text") or row.get("display_text"))
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


def _claim_answer(claims) -> str:
    claim = claims[0]
    if claim.status == "contradicted":
        return f"Klaim “{claim.claim_text}” bertentangan dengan segmen terverifikasi."
    return f"Klaim “{claim.claim_text}” tidak didukung oleh segmen terverifikasi dalam korpus ini."


def _verified_answer_facts(evidence: tuple[dict, ...], fallback: str) -> dict[str, str]:
    """Expose only complete, verified fact sentences to the wording boundary."""
    facts = {"deterministic_answer": fallback}
    for row in evidence:
        evidence_id = str(row.get("evidence_id") or "")
        quote = _compact_text(row.get("quoted_text") or row.get("display_text"))
        citation = str(row.get("label") or row.get("citation") or "Bukti")
        if evidence_id and quote:
            facts[f"support:{evidence_id}"] = f"{citation}: {quote}"
    return facts


def _render_wording(proposal: object, fallback: str, facts: dict[str, str] | None = None) -> str:
    """Render only fact-bound wording; an external model never adds atoms."""
    if isinstance(proposal, dict) and set(proposal) == {"sentences"}:
        sentences = proposal.get("sentences")
        approved = facts or {"deterministic_answer": fallback}
        if not isinstance(sentences, tuple) or not sentences:
            return fallback
        rendered: list[str] = []
        for sentence in sentences:
            if not isinstance(sentence, dict):
                return fallback
            refs = sentence.get("referenced_fact_ids")
            text = sentence.get("text")
            if not isinstance(text, str) or not isinstance(refs, tuple) or not refs or not set(refs) <= set(approved):
                return fallback
            if any(unicodedata.category(char) in {"Cf", "Cc"} for char in text):
                return fallback
            source_tokens = Counter(re.findall(r"\w+", " ".join(approved[item] for item in refs).casefold(), flags=re.UNICODE))
            proposal_tokens = Counter(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
            if proposal_tokens != source_tokens:
                return fallback
            rendered.append(text.strip())
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


def _empty_citation_fields() -> dict:
    return {
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "citation_payloads": (),
        "viewer_refs": (),
        "validation_reasons": {},
    }


def _answer_templates(store) -> dict[str, str]:
    configured: dict = getattr(getattr(store, "config", None), "setting", lambda *args: {})("answer_templates", {})
    return _ANSWER_TEMPLATES | dict(configured or {})


def _clarification_response(routed: dict, decision) -> dict:
    options = tuple({"label": item.label, "resolution": item.resolution} for item in decision.options)
    return {
        "status": "clarification_required", "route": _ask_route(str(routed.get("route") or "")), "intent": routed.get("intent"),
        "reason": routed.get("reason") or "ambiguous_interpretation", "answer_type": "clarification", "answer": decision.question,
        "answer_scope": "clarification", "clarification_kind": decision.kind, "clarification_question": decision.question,
        "clarification_options": options,
        "context_pack": empty_context_pack(routed.get("reason") or "ambiguous_interpretation"), "evidence": (), "citations": (),
        "final_citations": (), "historical_citations": (), "metadata_support": (), "structural_support": (), "trace_support": (),
        "viewer_refs": (), "metadata_facts": (), "legal_relations": (), "warnings": ("clarification_required",),
        "insufficient_reasons": ("ambiguous_interpretation",),
    }


def _scope_has_verified_support(store, routed: dict) -> bool:
    return any(
        validate_answer_candidate(store, row)[0]
        for row in routed.get("matches", ())
    )


def _authority_policy(store, row: dict, *, can_resolve: bool | None = None, conflict: dict | None = None) -> dict:
    owner = store.get(row.get("evidence_id")) if store is not None and row.get("evidence_id") else None
    source_row = {**(owner or {}), **row}
    authority_kind = _authority_kind(store, row, can_resolve=can_resolve, conflict=conflict)
    conflict_row = conflict or _source_conflict_by_evidence(store, row.get("evidence_id"))
    non_final_conflict = conflict_row is not None or row.get("source_conflict_id")
    citation_final = row.get("citation_final") if isinstance(row.get("citation_final"), bool) else authority_kind == "legal_citation"
    if non_final_conflict and authority_kind in {"source_anomaly", "source_conflict_provenance"}:
        citation_final = False
    layout_lines = _layout_lines(store, source_row)
    copy_text, layout_lines = _canonical_text_projection(
        row.get("copy_text") or row.get("quoted_text") or "", layout_lines
    )
    payload = {
        "authority_kind": authority_kind,
        "authority_label": {
            "legal_citation": "Sitasi hukum",
            "metadata_source": "Metadata sumber",
            "metadata_trace": "Metadata trace",
            "source_conflict_provenance": "Jejak audit sumber",
            "source_anomaly": "Source anomaly",
            "structural_context": "Provenance struktural",
            "instrument_provenance": "Instrument provenance",
            "source_text": "Sumber teks PDF",
        }[authority_kind],
        "citation_final": citation_final,
        "source_url": row.get("source_url") or _source_url(store, row),
        "support_kind": "legal_unit" if source_row.get("evidence_owner_kind") == "legal_unit_source" and authority_kind == "legal_citation" else row.get("support_kind") or _support_kind_for_authority(authority_kind),
        "relevant_quote_eligible": source_row.get("relevant_quote_eligible") is True and authority_kind == "legal_citation",
        "display_text": row.get("display_text") or row.get("quoted_text") or "",
        "source_label": row.get("document_title") or _source_label(store, row),
        "copy_text": copy_text,
        "layout_lines": layout_lines,
        "viewer_target": _viewer_target(row),
    }
    if conflict_row is not None or row.get("source_conflict_id"):
        payload |= _source_conflict_taxonomy_fields(conflict_row or row)
    return payload


def _canonical_text_projection(value: str, layout_lines: tuple[dict, ...] = ()) -> tuple[str, tuple[dict, ...]]:
    """Build semantic copy text and visual-line ranges in one ordered traversal."""
    if layout_lines:
        pieces: list[str] = []
        projected: list[dict] = []
        current_id = object()
        for line in layout_lines:
            paragraph_id = line.get("paragraph_id")
            text = " ".join(str(line.get("text") or "").split())
            if not text:
                continue
            if pieces:
                pieces.append("\n\n" if paragraph_id != current_id else " ")
            current_id = paragraph_id
            start = sum(len(piece) for piece in pieces)
            pieces.append(text)
            projected.append(line | {"text": text, "canonical_start": start, "canonical_end": start + len(text)})
        if projected:
            return "".join(pieces), tuple(projected)
    text = "\n".join(line.lstrip(" \t") for line in str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()
    return text, tuple(layout_lines)


def _copy_text(value: str, layout_lines: tuple[dict, ...] = ()) -> str:
    return _canonical_text_projection(value, layout_lines)[0]


def _viewer_target(row: dict) -> dict:
    target = dict(row.get("viewer_target") or row.get("viewer_ref") or {})
    target.setdefault("source_document_id", row.get("source_document_id"))
    target.setdefault("evidence_id", row.get("evidence_id"))
    target.setdefault("page_numbers", tuple(row.get("page_numbers") or ()))
    return target


def _select_viewer_bboxes(bboxes: list[dict], bbox_refs: tuple[str, ...]) -> list[dict]:
    """Honor the verified support subset stored behind an opaque target."""
    if not bbox_refs:
        return bboxes
    expected = set(bbox_refs)
    selected = [bbox for bbox in bboxes if bbox.get("bbox_id") in expected]
    return selected if {bbox.get("bbox_id") for bbox in selected} == expected else []


def _layout_lines(store, row: dict) -> tuple[dict, ...]:
    """Expose source-derived line layout without making the UI infer it."""
    configured = row.get("layout_lines")
    if isinstance(configured, (list, tuple)) and configured and isinstance(configured[0], dict):
        return tuple(configured)
    spans = {
        item.get("text_span_id"): item
        for item in (getattr(store, "page_text_spans", ()) if store is not None else ())
    }
    fragments = []
    for order, span_id in enumerate(row.get("text_span_ids") or ()):
        span = spans.get(span_id)
        if not span:
            continue
        boxes = store.exact_bboxes_for_text_spans((span_id,)) if store is not None else ()
        geometry = boxes or (span,)
        width = next((float(box["page_width"]) for box in geometry if isinstance(box.get("page_width"), (int, float))), 0.0)
        x0 = min((float(box["x0"]) for box in geometry if isinstance(box.get("x0"), (int, float))), default=0.0)
        x1 = max((float(box["x1"]) for box in geometry if isinstance(box.get("x1"), (int, float))), default=0.0)
        y0 = min((float(box["y0"]) for box in geometry if isinstance(box.get("y0"), (int, float))), default=0.0)
        y1 = max((float(box["y1"]) for box in geometry if isinstance(box.get("y1"), (int, float))), default=0.0)
        fragments.append({
            "text": span.get("exact_quote") or span.get("text") or "",
            "order": order,
            "page": span.get("page_number"),
            "width": width,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "refs": [box["bbox_id"] for box in boxes if box.get("bbox_id")],
        })
    if fragments:
        # Page-text spans may be fragments of one visual source line.  Merge
        # only exact same-baseline fragments in extraction order.
        visual: list[dict[str, Any]] = []
        for fragment in fragments:
            previous = visual[-1] if visual else None
            previous_part = previous["parts"][-1] if previous is not None else None
            same_line = previous is not None and previous_part is not None and fragment["page"] == previous["page"] and abs(fragment["y0"] - previous_part["y0"]) <= 1.0 and abs(fragment["y1"] - previous_part["y1"]) <= 1.0
            if same_line and previous is not None:
                previous["parts"].append(fragment)
            else:
                visual.append({"page": fragment["page"], "parts": [fragment]})
        page_left = {
            page: min(part["x0"] for line in visual if line["page"] == page for part in line["parts"])
            for page in {line["page"] for line in visual}
        }
        result = []
        paragraph = 0
        previous = None
        for line_order, line in enumerate(visual):
            parts = line["parts"]
            x0, x1 = min(part["x0"] for part in parts), max(part["x1"] for part in parts)
            y0, y1 = min(part["y0"] for part in parts), max(part["y1"] for part in parts)
            width = next((part["width"] for part in parts if part["width"]), 0.0)
            text = " ".join(str(part["text"]).strip() for part in parts if str(part["text"]).strip())
            centered = width and (x1 - x0) <= width * 0.55 and abs(((x0 + x1) / 2) - width / 2) <= max(8.0, width * 0.04)
            alignment = "center" if centered else "left"
            # A numbered line is a semantic boundary even when its baseline
            # follows the prior line closely (for example consecutive ayat).
            numbered = bool(re.match(r"^\s*(?:\(\d+\)|[A-Za-z]\.|\d+[.)])\s+", text))
            if previous and (
                line["page"] != previous["page"]
                or y0 - previous["y1"] > max(20.0, 1.5 * (y1 - y0))
                or alignment == "center"
                or previous["alignment"] == "center"
                or numbered
            ):
                paragraph += 1
            result.append({
                "text": text,
                "line_order": line_order,
                "paragraph_id": str(paragraph),
                "alignment": alignment,
                "indent": max(0.0, x0 - page_left[line["page"]]) if alignment == "left" else 0.0,
                "source_bbox_refs": [ref for part in parts for ref in part["refs"]],
            })
            previous = {"page": line["page"], "y1": y1, "alignment": alignment}
        return tuple(result)
    text = str(row.get("display_text") or row.get("quoted_text") or "")
    return ({"text": text, "line_order": 0, "paragraph_id": str(row.get("evidence_id") or "support"), "alignment": "unknown", "indent": 0.0, "source_bbox_refs": []},)


def _support_kind_for_authority(authority_kind: str) -> str:
    return {
        "legal_citation": "legal_citation",
        "metadata_source": "metadata_source",
        "metadata_trace": "metadata_trace",
        "source_conflict_provenance": "source_anomaly_provenance",
        "source_anomaly": "source_anomaly_provenance",
        "structural_context": "structural_provenance",
        "instrument_provenance": "instrument_provenance",
        "source_text": "source_text",
    }[authority_kind]


def _source_url(store, row: dict) -> str | None:
    source_id = row.get("source_document_id")
    return next(
        (
            source.get("source_page_url") or source.get("final_download_url") or source.get("download_url")
            for source in getattr(store, "source_documents", ())
            if source.get("source_document_id") == source_id
        ),
        None,
    )


def _source_label(store, row: dict) -> str | None:
    source_id = row.get("source_document_id")
    source: dict[str, Any] = next((item for item in getattr(store, "source_documents", ()) if item.get("source_document_id") == source_id), {})
    catalog: dict[str, Any] = getattr(getattr(store, "config", None), "setting", lambda *args: {})("document_catalog", {}) or {}
    return (catalog.get("titles") or {}).get(source.get("source_role")) or source.get("filename")


def _source_conflict_taxonomy_fields(conflict: dict | None) -> dict:
    if not conflict:
        return {}
    policy = conflict.get("source_anomaly_policy") or {}
    fields = {
        "source_anomaly_kind": conflict.get("source_anomaly_kind") or policy.get("anomaly_kind"),
        "source_mapping_kind": conflict.get("source_mapping_kind") or policy.get("mapping_kind"),
        "provenance_highlight_scope": conflict.get("provenance_highlight_scope") or policy.get("provenance_highlight_scope"),
        "finality_policy": policy.get("finality_policy"),
        "support_type": conflict.get("type"),
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    if fields.get("finality_policy"):
        fields["support_kind"] = fields["finality_policy"]
    fields.update(_source_mapping_semantics(conflict))
    return fields


def _authority_kind(store, row: dict, *, can_resolve: bool | None = None, conflict: dict | None = None) -> str:
    viewer_resolvable = (
        can_resolve
        if can_resolve is not None
        else row.get("viewer_ref", {}).get("can_resolve") is True or row.get("viewer_highlightable") is True
    )
    if row.get("metadata_grounding") or row.get("metadata_field"):
        return "metadata_source" if viewer_resolvable else "metadata_trace"
    if row.get("evidence_owner_kind") == "source_span" or row.get("authority_kind") == "source_text":
        return "source_text"
    if row.get("authority_kind") == "source_anomaly_trace" and row.get("citation_final") is False:
        return "source_anomaly"
    if row.get("presentation_as_legal_quote") is True:
        return "legal_citation"
    if row.get("authority_kind") == "structural_context":
        return "structural_context"
    conflict_row = conflict or _source_conflict_by_evidence(store, row.get("evidence_id"))
    if conflict_row is not None or row.get("source_conflict_id"):
        return "source_anomaly" if _is_source_anomaly_conflict(conflict_row or row) else "source_conflict_provenance"
    if _row_is_historical_anomaly(store, row):
        return "source_anomaly"
    if _row_is_instrument_provenance(store, row):
        return "instrument_provenance"
    return "legal_citation"


def _source_conflict_by_evidence(store, evidence_id: object) -> dict | None:
    if store is None or not evidence_id:
        return None
    return next(
        (
            row
            for row in store.source_conflicts
            if evidence_id == row.get("source_conflict_id") or evidence_id in set(row.get("evidence_ids") or ())
        ),
        None,
    )


def _is_source_anomaly_conflict(row: dict) -> bool:
    if row.get("source_anomaly_kind") == "renumbering_provenance":
        return False
    if row.get("source_anomaly_kind") == "source_marker_sequence_anomaly":
        return True
    classification = str(row.get("classification") or "").casefold()
    return (
        row.get("provenance_exception_category") == "accepted_noncanonical_source_conflict_trace_only"
        or "anomaly" in classification
        or "typo" in classification
    )


def _row_is_instrument_provenance(store, row: dict) -> bool:
    candidate_type = str(row.get("candidate_type") or "")
    if candidate_type == "article_amendment_relation" or candidate_type.startswith("instrument_"):
        return True
    if store is None:
        return False
    try:
        units = store.legal_units
    except (KeyError, OSError, ValueError):
        return False
    unit: dict = next((item for item in units if item.get("legal_unit_id") == row.get("legal_unit_id")), {})
    return bool(unit) and _is_instrument_unit(store, unit)


def _row_is_historical_anomaly(store, row: dict) -> bool:
    if store is None:
        return False
    try:
        units = store.legal_units
    except (KeyError, OSError, ValueError):
        return False
    unit: dict = next((item for item in units if item.get("legal_unit_id") == row.get("legal_unit_id")), {})
    return unit.get("status") == "active_historical_record" and bool(unit.get("exclusion_ref"))


def _citation_with_authority(store, row: dict, *, conflict: dict | None = None) -> dict:
    return row | _authority_policy(store, row, conflict=conflict)


def _claim_citations(citations: tuple[dict, ...], claim_support) -> tuple[dict, ...]:
    segments = tuple(
        segment
        for claim in claim_support
        for segment in claim.support_segments
        if segment.get("evidence_id")
    )
    return tuple(
        _claim_citation(
            row,
            next(
                (
                    segment
                    for segment in segments
                    if segment.get("source_document_id") == row.get("source_document_id")
                    and (
                        segment.get("evidence_id") == row.get("evidence_id")
                        or segment.get("legal_unit_id") == row.get("legal_unit_id")
                    )
                ),
                None,
            ),
        )
        for row in citations
    )


def _claim_citation(citation: dict, segment: dict | None) -> dict:
    if segment is None:
        return citation
    bbox_refs = tuple(segment.get("bbox_refs") or ())
    exact_quote = segment.get("exact_quote")
    return citation | {
        "proposition_id": segment.get("proposition_id"),
        "quoted_text": exact_quote,
        "display_text": exact_quote,
        "copy_text": exact_quote,
        "layout_lines": (),
        "text_span_ids": tuple(segment.get("text_span_ids") or ()),
        "bbox_refs": bbox_refs,
        "page_numbers": tuple(segment.get("page_numbers") or ()),
        "bbox_count": len(bbox_refs),
        "viewer_overlay": segment.get("viewer_overlay"),
        "viewer_ref": {
            **dict(citation.get("viewer_ref") or {}),
            "bbox_count": len(bbox_refs),
            "can_resolve": bool(bbox_refs),
        },
    }


def _source_span_evidence(store, support_id: str) -> dict | None:
    span = store.source_span_for_support(support_id)
    if not span or not span.get("semantic_text") or span.get("citation_eligible") is not True:
        return None
    if not store.source_span_bboxes(support_id):
        return None
    return {
        "corpus_id": getattr(store.config, "corpus_id", None),
        "evidence_id": support_id,
        "source_support_id": support_id,
        "source_document_id": span.get("source_document_id"),
        "source_pdf_path": span.get("source_pdf_path"),
        "source_sha256": span.get("source_sha256"),
        "source_role": span.get("source_role"),
        "temporal_context": span.get("source_role"),
        "citation": span.get("semantic_exact_quote"),
        "quoted_text": span.get("semantic_exact_quote"),
        "page_numbers": [span.get("page_number")],
        "bbox_refs": [support_id],
        "bbox_precision": "exact",
        "viewer_highlightable": True,
        "status": "final",
        "citation_final": False,
        "citation_eligible": True,
        "relevant_quote_eligible": False,
        "authority_kind": "source_text",
        "support_kind": "source_text",
        "evidence_owner_kind": "source_span",
        "text_span_ids": (),
    }


def _viewer_with_authority(store, evidence: dict, payload: dict) -> dict:
    return payload | _authority_policy(store, evidence, can_resolve=payload.get("viewer_highlightable") is True)


def _catalog_search(store, corpus_id: str, query: str, limit: int, filters: dict) -> tuple[dict, ...]:
    query_text = normalize_intent_text(query)
    if not query_text:
        return ()
    catalog = store.config.setting("document_catalog", {}) or {}
    if not contains_intent_phrase(query, tuple(catalog.get("document_terms") or ())):
        return ()
    rows = [
        _document_result(store, corpus_id, source, _document_search_score(store, source, query_text))
        for source in store.source_documents
        if _document_matches_filters(source, filters)
    ]
    rows = [row for row in rows if row["_score"] > 0]
    rows.sort(key=lambda row: (-row["_score"], row["title"]))
    return tuple(({key: value for key, value in row.items() if key != "_score"}) for row in rows[:limit])


def _document_search_score(store, source: dict, query_text: str) -> int:
    haystack = normalize_intent_text(
        " ".join(
            str(item or "")
            for item in (
                _document_title(store, source),
                source.get("filename"),
                source.get("source_role"),
                source.get("temporal_context"),
            )
        )
    )
    return sum(1 for token in query_text.split() if token in haystack)


def _document_result(store, corpus_id: str, source: dict, score: int) -> dict:
    page_count = int(source.get("page_count") or 0)
    return {
        "_score": score,
        "corpus_id": corpus_id,
        "source_document_id": source.get("source_document_id"),
        "document_id": source.get("source_document_id"),
        "title": _document_title(store, source),
        "document_title": _document_title(store, source),
        "snippet": f"Dokumen sumber terverifikasi: {source.get('filename')} ({page_count} halaman).",
        "source_url": source.get("source_page_url") or source.get("final_download_url") or source.get("download_url"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
        "page_numbers": (1,),
        "bbox_count": 0,
        "viewer_ref": {
            "action": "viewer",
            "source_document_id": source.get("source_document_id"),
            "page_numbers": (1,),
            "bbox_count": 0,
            "can_resolve": True,
        },
        "status": "document",
    }


def _document_title(store, source: dict) -> str:
    catalog = store.config.setting("document_catalog", {}) or {}
    return (catalog.get("titles") or {}).get(source.get("source_role")) or source.get("filename") or source["source_document_id"]


def _document_matches_filters(source: dict, filters: dict) -> bool:
    return all(
        source.get(key) == value for key, value in filters.items() if key in {"source_role", "temporal_context"} and value is not None
    )


def _catalog_search_response(
    corpus_id: str,
    query: str,
    rows: tuple[dict, ...],
    status: str,
    reason: str | None,
    *,
    applied_filters: dict | None = None,
    invalid_filters: tuple[str, ...] = (),
) -> dict:
    return {
        "status": status,
        "public_status": "found" if rows else status,
        "route": "document_catalog" if status != "unsupported_corpus" else "unsupported_corpus",
        "intent": "document_catalog_search",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "matches": (),
        "reason": reason,
        "required_corpus": None,
        "applied_filters": applied_filters or {},
        "invalid_filters": invalid_filters,
        "results": rows,
        "context_pack": empty_context_pack(reason),
    }


def _search_result(row: dict, routed: dict, context_pack: dict) -> dict:
    viewer_ref = row.get("viewer_ref") or {}
    return {
        "corpus_id": routed["corpus_id"],
        "legal_unit_id": row.get("legal_unit_id"),
        "evidence_id": row["evidence_id"],
        "citation_id": row["evidence_id"],
        "viewer_ref_id": viewer_ref.get("evidence_id"),
        "source_document_id": row.get("source_document_id"),
        "title": row.get("label") or row.get("citation") or row.get("legal_unit_id") or routed["corpus_id"].upper(),
        "snippet": row.get("quoted_text"),
        "document_title": row.get("document_title"),
        "source_url": row.get("source_url"),
        "citation": row.get("citation"),
        "label": row.get("label"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": viewer_ref,
        "retrieval_method": routed["route"],
        "reasons": context_pack["validation_reasons"].get(row["evidence_id"]),
        "status": "evidence",
    }


def _ask_route(route: str) -> str:
    return {
        "exact": "legal_reference",
        "structured": "legal_reference",
        "structural_navigation": "structural_navigation",
        "structure_list": "structural_navigation",
        "metadata": "metadata_fact",
        "metadata_not_found": "metadata_fact",
        "metadata_scope_unresolved": "metadata_fact",
        "relation": "legal_relation",
        "relation_not_found": "legal_relation",
        "citation_not_found": "legal_reference",
        "structured_not_found": "legal_reference",
        "scope_unresolved": "legal_reference",
        "bm25": "lexical_fallback",
        "hybrid": "lexical_fallback",
        "hybrid_degraded_sparse": "lexical_fallback",
    }.get(route, route)


def _answer_type(route: str, status: str) -> str:
    if status != "answer_ready":
        return "limited_evidence_summary"
    return {
        "metadata_fact": "metadata_fact",
        "legal_relation": "legal_relation",
    }.get(route, "quoted_evidence")


def _unique_printed_names(rows: tuple[dict, ...]) -> tuple[str, ...]:
    """Collapse formatting variants without discarding their source supports."""
    names: dict[str, str] = {}
    for row in rows:
        value = str(row.get("printed_name") or "").strip()
        if row.get("fact_kind") != "person_role" or not value:
            continue
        names.setdefault(normalize_intent_text(value), value)
    return tuple(names.values())


def _metadata_fact(row: dict) -> dict:
    return {
        "field": row.get("metadata_field"),
        "answer": row.get("metadata_answer"),
        "evidence_id": row.get("evidence_id"),
    }


def _metadata_support(store, row: dict) -> dict:
    can_resolve = row.get("viewer_ref", {}).get("can_resolve") is True
    authority = _authority_policy(store, row, can_resolve=can_resolve)
    viewer_ref = ((row.get("viewer_ref") or {}) | authority) if can_resolve else None
    field = str(row.get("metadata_field") or "")
    labels = {
        "signatories": "Penandatangan",
        "penetapan": "Tanggal Penetapan",
        "institution": "Lembaga",
        "place": "Tempat Penetapan",
    }
    source = _source_document_meta(store, row.get("source_document_id"))
    document_title = _document_title(store, source)
    name = str(row.get("printed_name") or "").strip()
    role = str(row.get("printed_role") or "").strip()
    institution = str(row.get("institution") or "").strip()
    date_context = str(row.get("date_context") or "").strip()
    display_text = str(row.get("display_text") or row.get("answer") or "").strip()
    if name and role:
        display_text = f"{name} tercantum sebagai {role} dalam {document_title}."
    return authority | {
        "support_class": "exact_metadata_citation" if can_resolve else "metadata_trace",
        "field": row.get("metadata_field"),
        "answer": row.get("metadata_answer"),
        "fact_kind": row.get("fact_kind") or ("source_fact" if field else "metadata"),
        "display_label": labels.get(field, "Sumber Dokumen"),
        "display_text": display_text,
        "copy_text": _copy_text(display_text),
        "printed_name": name or None,
        "entity_identity": row.get("entity_identity"),
        "printed_name_alias": row.get("printed_name_alias"),
        "printed_role": role or None,
        "institution": institution or None,
        "date_context": date_context or None,
        "evidence_id": row.get("evidence_id"),
        "source_document_id": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "citation_available": can_resolve,
        "viewer_highlightable": can_resolve,
        "viewer_ref": viewer_ref,
    }


def _metadata_grounding_evidence(store, metadata_grounding_id: str | None) -> dict | None:
    for row in store.metadata_grounding:
        if row.get("metadata_grounding_id") == metadata_grounding_id:
            return {
                "evidence_id": row["metadata_grounding_id"],
                "metadata_grounding": True,
                "legal_unit_id": None,
                "citation": f"Metadata {row.get('source_role')}: {row.get('metadata_field') or 'block'}",
                "hierarchy": (),
                "quoted_text": row.get("quoted_text"),
                "bbox_refs": tuple(row.get("bbox_refs") or ()),
                "bbox_precision": row.get("bbox_precision"),
                "viewer_highlightable": row.get("viewer_highlightable"),
                "page_numbers": tuple(row.get("page_numbers") or ()),
                "source_document_id": row.get("source_document_id"),
                "source_pdf_path": row.get("source_pdf_path"),
                "source_sha256": row.get("source_sha256"),
                "source_role": row.get("source_role"),
                "temporal_context": row.get("temporal_context"),
            }
    return None


def _relation_response(store, routed: dict) -> dict:
    templates = _answer_templates(store)
    target = routed.get("relation_target") or {"mode": None}
    graph_edges = tuple(routed.get("matches") or ())
    if target["mode"] == "article":
        return _project_article_relation(store, routed, target, templates, _relation_support(graph_edges, "article_amendment_relation_graph"))
    if target["mode"] == "unsupported":
        return _relation_not_promoted(routed, templates)
    support = _relation_support(graph_edges, "document_relation_graph")
    if not support:
        reason = "document_relation_not_found"
        return project_response(
            routed | {"matches": (), "reason": reason},
            AnswerDecision(
                "insufficient_evidence",
                "document_relation",
                "none",
                templates["insufficient"],
                empty_context_pack(reason),
                document_relations=(),
                insufficient_reasons=(reason,),
            ),
        )
    relations = tuple(_public_document_relation(row) for row in support)
    # Document-level graph edges are provenance traces, not publishable legal
    # support. Keep the relation available for audit/UI context but never
    # promote a trace-only result to an answer-ready publication.
    return project_response(
        routed | {"matches": support},
        AnswerDecision(
            "limited_answer",
            "document_relation",
            "document_relation",
            _document_relation_answer(store, relations),
            empty_context_pack("document_relation_source_role_trace"),
            document_relations=relations,
            trace_support=relations,
            answer_scope="source_role_document_relation",
            warnings=("document_relation_not_exact_highlightable", "document_relation_trace_only"),
        ),
    )


def _project_article_relation(store, routed: dict, target: dict, templates: dict[str, str], support: tuple[dict, ...]) -> dict:
    if not support:
        if target.get("target_citation"):
            return _relation_not_promoted(routed, templates, reason="relation_target_not_found")
        return _relation_not_promoted(routed, templates)
    exact_support = tuple(row for row in support if _is_exact_article_relation(row))
    exact_targets = {row.get("target_legal_unit_id") for row in exact_support}
    trace_support = tuple(row for row in support if not _is_exact_article_relation(row) and row.get("target_legal_unit_id") not in exact_targets)
    requested_targets = {
        _normalize_relation_citation(value)
        for value in target.get("target_citations") or ()
        if value
    }
    exact_citations = {
        _normalize_relation_citation(row.get("target_citation") or row.get("target_reference"))
        for row in exact_support
    }
    if requested_targets - exact_citations:
        trace_support = tuple(
            row
            for row in support
            if _normalize_relation_citation(row.get("target_citation") or row.get("target_reference")) in requested_targets - exact_citations
        )
    # Exact source relations are the publishable article targets. Trace rows
    # remain available for the trace-only path, but must not be projected as
    # neighboring article answers when an exact target already satisfies the
    # request.
    public_relations = tuple(
        _public_article_relation(row) for row in (exact_support if exact_support else trace_support)
    )
    answer_evidence = tuple(row for row in (_article_relation_evidence(store, row) for row in exact_support) if row)
    if not answer_evidence:
        if not trace_support:
            return _relation_not_promoted(routed, templates)
        return project_response(
            routed | {"matches": support, "reason": "relation_trace_only"},
            AnswerDecision(
                "limited_answer",
                "document_relation",
                "article_amendment_relation",
                _article_relation_answer(store, (), trace_support),
                empty_context_pack("relation_trace_only"),
                article_amendment_relations=public_relations,
                relation_support=(),
                trace_support=tuple(_public_article_relation(row) for row in trace_support),
                answer_scope="trace_article_relation",
                warnings=("article_relation_trace_only_not_citable",),
            ),
        )
    citations = _deduplicated_article_relation_citations(store, answer_evidence)
    final_citations = tuple(row for row in citations if row.get("citation_final") is True)
    historical_citations = tuple(row for row in citations if row.get("citation_final") is False)
    viewer_refs = tuple(row["viewer_ref"] for row in answer_evidence if row.get("citation_final") is True)
    public_trace_support = trace_support
    partial = bool(public_trace_support)
    public_evidence = answer_evidence
    context_pack = {
        "answer_evidence": public_evidence,
        "supporting_context": (),
        "excluded_results": (),
        "citation_payloads": final_citations,
        "historical_citations": historical_citations,
        "viewer_refs": viewer_refs,
        "validation_reasons": {row["evidence_id"]: "article_amendment_relation_exact_source_text" for row in public_evidence},
    }
    return project_response(
        routed | {"matches": support},
        AnswerDecision(
            "limited_answer" if partial else "answer_ready",
            "document_relation",
            "article_amendment_relation",
            _article_relation_answer(store, exact_support, public_trace_support),
            context_pack,
            evidence=public_evidence,
            citations=final_citations,
            final_citations=final_citations,
            historical_citations=historical_citations,
            viewer_refs=viewer_refs,
            article_amendment_relations=public_relations,
            relation_support=answer_evidence,
            trace_support=tuple(_public_article_relation(row) for row in public_trace_support),
            answer_scope="partial_exact_article_relation" if partial else "exact_article_relation",
            warnings=("article_relation_exact_support_partial_trace_omitted",) if public_trace_support else (),
        ),
    )


def _normalize_relation_citation(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("(", "").replace(")", "").split())


def _relation_not_promoted(routed: dict, templates: dict[str, str], *, reason: str = "relation_not_promoted") -> dict:
    return project_response(
        routed | {"matches": (), "reason": reason},
        AnswerDecision(
            "insufficient_evidence",
            "document_relation",
            "none",
            templates["insufficient"],
            empty_context_pack(reason),
            document_relations=(),
            article_amendment_relations=(),
            insufficient_reasons=(reason,),
        ),
    )


def _relation_support(graph_edges: tuple[dict, ...], route_source: str) -> tuple[dict, ...]:
    return tuple(
        edge["relation_projection"] | {"route_sources": edge.get("route_sources") or (route_source,)}
        for edge in graph_edges
        if edge.get("relation_projection")
    )


def _article_relation_evidence(store, relation: dict) -> dict | None:
    if not _is_exact_article_relation(relation):
        return None
    row = store.get(relation["evidence_id"])
    if row is None:
        return None
    if store.lineage_error(row):
        return None
    proof_bbox_refs = tuple(relation.get("bbox_refs") or row.get("bbox_refs") or ())
    proof_text_span_ids = tuple(relation.get("text_span_ids") or row.get("text_span_ids") or ())
    source_quote = _source_quote_for_spans(store, proof_text_span_ids)
    target_bbox_refs = tuple(relation.get("target_bbox_refs") or ())
    proof_bboxes = store.bboxes_for_refs(proof_bbox_refs)
    if not proof_bboxes or not set(proof_bbox_refs) <= {bbox["bbox_id"] for bbox in proof_bboxes}:
        return None
    if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True:
        return None
    return {
        **row,
        "relation_id": relation.get("relation_id"),
        "support_kind": "article_relation",
        "fact_kind": "article_relation",
        "display_label": relation.get("target_label") or relation.get("target_citation") or relation.get("relation_type") or "Relasi Pasal",
        "display_text": source_quote or relation.get("quoted_text") or row.get("quoted_text"),
        "bbox_refs": proof_bbox_refs,
        "text_span_ids": proof_text_span_ids,
        "relation_source_proof_bbox_refs": proof_bbox_refs,
        "relation_source_proof_text_span_ids": proof_text_span_ids,
        "relation_target_bbox_refs": target_bbox_refs,
        "relation_target_text_span_ids": tuple(relation.get("target_text_span_ids") or ()),
        "quoted_text": source_quote or relation.get("quoted_text") or row.get("quoted_text"),
        "bbox_count": len(proof_bboxes),
        "route_sources": ("article_amendment_relation",),
        "article_amendment_relation": relation,
        "viewer_ref": {
            "action": "viewer",
            "evidence_id": row["evidence_id"],
            "relation_id": relation.get("relation_id"),
            "source_document_id": row.get("source_document_id"),
            "page_numbers": tuple(row.get("page_numbers") or ()),
            "text_span_ids": proof_text_span_ids,
            "bbox_count": len(proof_bboxes),
            "bbox_refs": proof_bbox_refs,
            "source_proof_text_span_ids": proof_text_span_ids,
            "source_proof_bbox_refs": proof_bbox_refs,
            "target_text_span_ids": tuple(relation.get("target_text_span_ids") or ()),
            "target_bbox_refs": target_bbox_refs,
            "can_resolve": True,
        },
    }


def _source_quote_for_spans(store, text_span_ids: tuple[str, ...]) -> str:
    by_id = {span.get("text_span_id"): span for span in store.page_text_spans}
    quotes = [str(by_id[span_id].get("exact_quote") or "") for span_id in text_span_ids if span_id in by_id]
    return "\n".join(quote for quote in quotes if quote)


def _relation_for_evidence(store, evidence_id: str | None, relation_id: str | None = None) -> dict | None:
    if not evidence_id:
        return None
    return next(
        (
            projection
            for edge in store.graph_edges
            for projection in (edge.get("relation_projection") or {},)
            if projection.get("target_legal_unit_id")
            and projection.get("evidence_id") == evidence_id
            and (relation_id is None or projection.get("relation_id") == relation_id)
        ),
        None,
    )


def _is_exact_article_relation(row: dict) -> bool:
    return (
        row.get("support_class") == "exact_article_relation"
        and row.get("grounding_level") == "exact_source_text"
        and row.get("bbox_precision") == "exact"
        and row.get("viewer_highlightable") is True
        and row.get("citation_available") is True
    )


def _public_document_relation(row: dict) -> dict:
    return {
        "relation_id": row.get("relation_id"),
        "relation_type": row.get("relation_type"),
        "source_document_id": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "target_source_role": row.get("target_source_role"),
        "target_document_id": row.get("target_document_id"),
        "support_type": row.get("support_type"),
        "reason": row.get("reason"),
        "highlightable": row.get("viewer_highlightable") is True,
    }


def public_article_relation(row: dict) -> dict:
    inverse = row.get("projection_direction") == "inverse"
    return {
        "relation_id": row.get("relation_id"),
        "relation_type": row.get("relation_type"),
        "source_document_id": row.get("support_document_id") or row.get("source_document_id"),
        "source_role": row.get("support_source_role") or row.get("source_role"),
        "source_legal_unit_id": row.get("source_legal_unit_id"),
        "source_legal_unit_role": row.get("source_legal_unit_role"),
        "source_label": row.get("source_label"),
        "source_reference": row.get("new_reference" if inverse else "old_reference") or row.get("source_reference"),
        "source_reference_range": row.get("new_reference_range" if inverse else "old_reference_range")
        or row.get("source_reference_range"),
        "source_reference_range_kind": row.get("new_reference_range_kind" if inverse else "old_reference_range_kind")
        or row.get("source_reference_range_kind"),
        "target_legal_unit_id": row.get("target_legal_unit_id"),
        "target_label": row.get("target_label") or row.get("target_citation"),
        "target_citation": row.get("target_citation"),
        "target_reference": row.get("old_reference" if inverse else "new_reference") or row.get("target_reference"),
        "target_reference_range": row.get("old_reference_range" if inverse else "new_reference_range")
        or row.get("target_reference_range"),
        "target_reference_range_kind": row.get("old_reference_range_kind" if inverse else "new_reference_range_kind")
        or row.get("target_reference_range_kind"),
        "target_source_role": row.get("target_source_role"),
        "evidence_id": row.get("evidence_id"),
        "bbox_refs": tuple(row.get("bbox_refs") or ()),
        "source_proof_text_span_ids": tuple(row.get("text_span_ids") or ()),
        "source_proof_bbox_refs": tuple(row.get("bbox_refs") or ()),
        "target_bbox_refs": tuple(row.get("target_bbox_refs") or ()),
        "target_precision": row.get("target_precision"),
        "source_support_exact": row.get("source_support_exact") is True,
        "text_span_ids": tuple(row.get("text_span_ids") or ()),
        "target_text_span_ids": tuple(row.get("target_text_span_ids") or ()),
        "support_class": row.get("support_class"),
        "grounding_level": row.get("grounding_level"),
        "authority_kind": row.get("authority_kind"),
        "citation_final": row.get("citation_final") is True,
        "recovery_capability": row.get("recovery_capability"),
        "recovery_status": row.get("recovery_status"),
        "target_geometry_method": row.get("target_geometry_method"),
        "trace_only_reason": row.get("trace_only_reason"),
        "citation_available": row.get("citation_available") is True,
        "viewer_highlightable": row.get("viewer_highlightable") is True,
    }


_public_article_relation = public_article_relation


def _article_relation_citation(row: dict) -> dict:
    return {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": row["evidence_id"],
        "legal_unit_id": row.get("legal_unit_id"),
        "source_document_id": row.get("source_document_id"),
        "citation": row.get("citation"),
        "label": row.get("citation"),
        "hierarchy": tuple(row.get("hierarchy") or ()),
        "quoted_text": row.get("quoted_text"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": row.get("viewer_ref"),
        "evidence_status": row.get("status"),
    }


def _deduplicated_article_relation_citations(store, rows: tuple[dict, ...]) -> tuple[dict, ...]:
    grouped: dict[object, dict] = {}
    for row in rows:
        citation = _article_relation_citation(row)
        key = citation.get("evidence_id")
        grouped.setdefault(key, citation)
    return tuple(_citation_with_authority(store, row) for row in grouped.values())


def _document_relation_answer(store, relations: tuple[dict, ...]) -> str:
    intent = store.config.setting("intent_config", {}) or {}
    relation_config = intent.get("document_relation", {}) or {}
    labels = intent.get("source_role_labels", {}) or {}
    prefix = str(relation_config.get("source_role_label_prefix", ""))
    amendment_roles = [_document_relation_amendment_role(row) for row in relations]
    amendment_roles = [role for role in amendment_roles if role]
    names = [f"{prefix}{labels.get(role, role)}" for role in amendment_roles]
    if len(names) > 1:
        listed = ", ".join(names[:-1]) + f", dan {names[-1]}"
        return str(relation_config.get("document_answer_template", "{relations}")).format(relations=listed)
    name = names[0] if names else "Perubahan"
    return str(relation_config.get("single_document_answer_template", "{relation}")).format(relation=name)


def _document_relation_amendment_role(row: dict) -> str | None:
    for key in ("source_role", "target_source_role"):
        role = str(row.get(key) or "")
        if role.startswith("amendment_"):
            return role
    return None


def _article_relation_answer(store, relations: tuple[dict, ...], trace_support: tuple[dict, ...]) -> str:
    def labels_for(rows: tuple[dict, ...]) -> list[str]:
        by_target: dict[str, set[str]] = {}
        for row in rows:
            target = str(row.get("new_reference") or row.get("target_citation") or "")
            if target:
                by_target.setdefault(target, set()).add(str(row.get("relation_type") or ""))
        labels = []
        for target in sorted(by_target, key=_legal_reference_sort_key):
            types = by_target[target]
            suffix = " / ".join(relation_labels[relation] for relation in ("DELETES", "MODIFIES", "RENAMES", "RENUMBERED_TO") if relation in types)
            labels.append(f"{target} ({suffix})" if suffix else target)
        return labels

    relation_labels = {
        "DELETES": "dihapus",
        "MODIFIES": "diubah",
        "RENAMES": "dinomori ulang",
        "RENUMBERED_TO": "dinomori ulang",
    }
    exact_labels = labels_for(tuple(relations))
    trace_labels = labels_for(tuple(trace_support))
    if not exact_labels and not trace_labels:
        return "Sumber terverifikasi tidak memuat relasi pasal yang dapat dipublikasikan."
    source_label = next(
        (str(row.get("source_label") or "").removesuffix(" Scope") for row in (*relations, *trace_support) if row.get("source_label")),
        "Sumber perubahan",
    )
    if exact_labels and trace_labels:
        return (
            f"{source_label} secara terverifikasi memuat perubahan pada {', '.join(exact_labels)}. "
            f"Keterbatasan: {', '.join(trace_labels)} hanya tersedia sebagai jejak sumber."
        )
    if trace_labels:
        return f"{source_label} menyebut {', '.join(trace_labels)}, tetapi dukungan yang tersedia hanya berupa jejak sumber."
    return f"{source_label} secara terverifikasi mengubah {', '.join(exact_labels)}."


def _legal_reference_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"Pasal\s*(\d+)([A-Za-z]?)", value)
    if not match:
        return (10**9, value.casefold())
    return (int(match.group(1)), match.group(2).casefold())


def _instrument_intent_context(store, query: str) -> tuple[dict | None, str, str] | None:
    if store is None:
        return None
    config = getattr(store, "config", None)
    intent = intent_config_for(getattr(config, "structured_strategy", "generic"), config)
    decision = resolve_instrument_intent(query, intent, corpus=getattr(config, "corpus_id", ""))
    if metadata_lookup(store, query, 1) and decision.reason not in {"analysis_metadata_conflict", "unsupported_analysis_intent"}:
        return None
    if decision.target_status == "not_instrument":
        return None
    if decision.target_status == "instrument_unresolved":
        return None, "instrument_unresolved", decision.reason
    row = next(
        (
            item
            for item in store.evidence
            if item.get("source_role") == decision.amendment and item.get("citation") == decision.target_citation
        ),
        None,
    )
    if row is None:
        return None, "instrument_unresolved", decision.reason
    row = row | {
        "route_sources": ("structured",),
        "candidate_type": f"instrument_{decision.role_family}_candidate",
    }
    accepted, _ = validate_answer_candidate(store, row)
    if accepted:
        return row, "instrument_resolved_answerable", "answer_evidence"
    return (
        row | {"forced_rejection_reason": "instrument_resolved_fail_closed"},
        "instrument_resolved_fail_closed",
        "instrument_resolved_fail_closed",
    )


def _is_instrument_unit(store, unit: dict) -> bool:
    schema: dict = getattr(getattr(store, "config", None), "setting", lambda *args: {})("schema", {}) or {}
    return unit.get("unit_type") in set(schema.get("instrument_unit_types") or ())


def _source_anomaly_response(store, corpus_id: str, query: str) -> dict | None:
    if store is None:
        return None
    if not _is_source_anomaly_query(store, query):
        return None
    conflict = _matched_source_conflict(store, query)
    if conflict is None:
        return _source_anomaly_fallback()
    support = _source_conflict_support(store, conflict)
    reasons = _source_conflict_reasons(store, query)
    exact_provenance = bool(support["evidence"])
    trace_only = bool(support["trace_support"]) and not exact_provenance
    answer = _source_anomaly_answer(store, conflict, query, exact_provenance=exact_provenance, trace_only=trace_only)
    return {
        "status": "limited_answer" if exact_provenance or trace_only else "insufficient_evidence",
        "route": "source_anomaly_explanation",
        "intent": "structured_lookup",
        "answer_type": "source_conflict_provenance" if exact_provenance or trace_only else "none",
        "answer": answer,
        "context_pack": support["context_pack"],
        "evidence": support["evidence"],
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "trace_support": support["trace_support"],
        "answer_scope": support["answer_scope"],
        "warnings": support["warnings"],
        "insufficient_reasons": tuple(dict.fromkeys(reasons)) if not exact_provenance else (),
        "source_conflict": _public_source_conflict(conflict),
    }


def _matched_source_conflict(store, query: str) -> dict | None:
    folded = (query or "").casefold()
    if not _is_source_anomaly_query(store, query):
        return None
    intent = _source_conflict_intent(store)
    matches = [(score, row) for row in store.source_conflicts if (score := _source_conflict_match_score(store, row, folded, intent)) > 0]
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1].get("source_conflict_id") or ""))
    return matches[0][1]


def _is_source_anomaly_query(store, query: str) -> bool:
    folded = (query or "").casefold()
    intent = _source_conflict_intent(store)
    terms = tuple(str(term).casefold() for term in intent.get("query_terms") or ())
    if not any(_query_contains_term(folded, term) for term in terms):
        return False
    unresolved_terms = tuple(str(term).casefold() for term in intent.get("unresolved_query_terms") or ())
    if any(_query_contains_term(folded, term) for term in unresolved_terms):
        return True
    discrepancy_markers = tuple(str(marker).casefold() for marker in intent.get("discrepancy_terms") or ())
    if any(_query_contains_term(folded, marker) for marker in discrepancy_markers):
        return any(
            sum(_query_contains_term(folded, str(anchor).casefold()) for anchor in conflict.get("query_anchor_terms") or ()) >= 2
            for conflict in store.source_conflicts
        )
    return False


def _source_anomaly_fallback() -> dict:
    return {
        "status": "insufficient_evidence",
        "route": "source_anomaly_explanation",
        "intent": "structured_lookup",
        "answer_type": "none",
        "answer": "Bukti tidak cukup untuk mengaitkan pertanyaan ini dengan catatan konflik sumber tertentu.",
        "context_pack": empty_context_pack("source_anomaly_unresolved"),
        "evidence": (),
        "citations": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "answer_scope": "insufficient_evidence",
        "warnings": (),
        "insufficient_reasons": ("source_anomaly", "source_anomaly_unresolved"),
        "source_conflict": None,
    }


def _source_conflict_reasons(store, query: str) -> list[str]:
    intent = _source_conflict_intent(store)
    folded = (query or "").casefold()
    reasons = ["source_anomaly", "canonical_conflict"]
    for rule in intent.get("reason_rules") or ():
        terms = tuple(str(term).casefold() for term in rule.get("query_terms") or ())
        if any(_query_contains_term(folded, term) for term in terms):
            reasons.extend(str(reason) for reason in rule.get("reasons") or ())
            return reasons
    reasons.extend(str(reason) for reason in intent.get("default_reasons") or ())
    return reasons


def _source_conflict_match_score(store, conflict: dict, folded_query: str, intent: dict) -> int:
    exclusions = tuple(str(term).casefold() for term in conflict.get("query_exclusion_terms") or ())
    if any(_query_contains_term(folded_query, term) for term in exclusions):
        return 0
    required = tuple(str(term).casefold() for term in conflict.get("query_required_terms") or ())
    anchors = {str(term).casefold() for term in conflict.get("query_anchor_terms") or ()}
    source_role = str(_source_document_meta(store, conflict.get("source_document_id")).get("source_role") or "")
    role_label = str((intent.get("role_labels") or {}).get(source_role) or source_role).casefold()
    role_anchor_match = _query_contains_term(folded_query, role_label) and any(
        _query_contains_term(folded_query, anchor) for anchor in anchors
    )
    natural_discrepancy = (
        any(_query_contains_term(folded_query, str(marker).casefold()) for marker in intent.get("discrepancy_terms") or ())
        and sum(_query_contains_term(folded_query, anchor) for anchor in anchors) >= 2
    )
    source_marker_context = conflict.get("source_anomaly_kind") in {"source_marker_sequence_anomaly", "typed_source_discrepancy"} and role_anchor_match
    if required and not any(_query_contains_term(folded_query, term) for term in required) and not source_marker_context and not natural_discrepancy:
        return 0
    explicit_anchor_match = any(
        len(anchor.split()) > 1 and _query_contains_term(folded_query, anchor) for anchor in anchors
    )
    semantic_required = tuple(term for term in required if term not in anchors or "pasal" not in term)
    marker_context = role_anchor_match or natural_discrepancy or any(_query_contains_term(folded_query, term) for term in semantic_required)
    if (
        semantic_required
        and not any(_query_contains_term(folded_query, term) for term in semantic_required)
        and not role_anchor_match
        and not (
            conflict.get("source_anomaly_kind") in {"source_marker_sequence_anomaly", "typed_source_discrepancy"}
            and (explicit_anchor_match or role_anchor_match)
        )
    ):
        return 0
    if conflict.get("source_anomaly_kind") in {"source_marker_sequence_anomaly", "typed_source_discrepancy"} and not marker_context:
        return 0
    score = 0
    if _query_contains_term(folded_query, role_label):
        score += 4
    for token in (str(value).casefold() for value in (conflict.get("query_anchor_terms") or conflict.get("anchor_terms") or ())):
        if _query_contains_term(folded_query, token):
            score += 3
    if score:
        return score
    haystack = (
        " ".join(
            str(value or "")
            for value in (
                conflict.get("source_conflict_id"),
                conflict.get("type"),
                conflict.get("source_anomaly_kind"),
                conflict.get("source_mapping_kind"),
                conflict.get("classification"),
                conflict.get("source_document_id"),
                role_label,
            )
        )
        .replace("_", " ")
        .casefold()
    )
    query_tokens = _meaningful_conflict_tokens(folded_query, intent)
    conflict_tokens = {token for token in re.findall(r"[a-z0-9]+", haystack) if len(token) > 2}
    overlap = query_tokens & conflict_tokens
    return len(overlap) if len(overlap) >= 2 else 0


def _query_contains_term(query: str, term: str) -> bool:
    """Match policy terms on token boundaries so Pasal suffixes cannot alias."""
    if not term:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", query) is not None


def _source_document_meta(store, source_document_id: object) -> dict:
    return next(
        (row for row in store.source_documents if row.get("source_document_id") == source_document_id),
        {},
    )


def _source_document_by_id(store, source_document_id: object) -> dict | None:
    return next((row for row in store.source_documents if row.get("source_document_id") == source_document_id), None)


def _meaningful_conflict_tokens(text: str, intent: dict) -> set[str]:
    generic = set(intent.get("generic_tokens") or ())
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in generic}


def _source_conflict_intent(store) -> dict:
    return store.config.setting("source_conflict_intent", {})


def _public_source_conflict(conflict: dict) -> dict:
    return {
        "source_conflict_id": conflict.get("source_conflict_id"),
        "type": conflict.get("type"),
        "classification": conflict.get("classification"),
        "source_document_id": conflict.get("source_document_id"),
        "status": conflict.get("status"),
    } | _source_conflict_contract_fields(conflict)


def _source_conflict_support(store, conflict: dict) -> dict:
    evidence_rows = tuple(
        row | {"route_sources": ("exact",), "candidate_type": "source_conflict_provenance"}
        for evidence_id in conflict.get("evidence_ids") or ()
        if (row := store.get(evidence_id)) is not None
    )
    context_pack = assemble_context_pack(store, evidence_rows) if evidence_rows else empty_context_pack("source_anomaly")
    evidence = context_pack["answer_evidence"]
    synthetic_support = _synthetic_source_conflict_support(store, conflict) if not evidence else None
    trace_support: tuple[dict, ...] = ()
    answer_scope = "insufficient_evidence"
    if evidence:
        trace_support = tuple(_citation_with_authority(store, row, conflict=conflict) for row in evidence)
        answer_scope = "source_conflict_exact_provenance"
    elif synthetic_support is not None:
        trace_support = tuple(synthetic_support["citations"])
        answer_scope = "source_conflict_exact_provenance"
    else:
        trace_support = (_source_conflict_trace_support(store, conflict, context_pack["validation_reasons"]),)
        answer_scope = "source_conflict_trace" if trace_support else "insufficient_evidence"
    viewer_refs = (
        tuple(
            ref | _authority_policy(store, evidence[0], can_resolve=ref.get("can_resolve") is True, conflict=conflict)
            for ref in context_pack["viewer_refs"]
        )
        if evidence
        else synthetic_support["viewer_refs"]
        if synthetic_support is not None
        else ()
    )
    citations = ()
    public_context_pack = (
        context_pack | {"citation_payloads": (), "viewer_refs": ()}
        if evidence
        else synthetic_support["context_pack"]
        if synthetic_support is not None
        else context_pack
    )
    return {
        "context_pack": public_context_pack,
        "evidence": synthetic_support["evidence"] if synthetic_support is not None else evidence,
        "citations": citations,
        "viewer_refs": viewer_refs,
        "trace_support": trace_support,
        "answer_scope": answer_scope,
        "warnings": (("source_conflict_not_final_legal_authority",) if evidence or synthetic_support is not None or trace_support else ()),
    }


def _source_conflict_trace_support(store, conflict: dict, validation_reasons: dict) -> dict:
    first_reason = next(iter(validation_reasons.values()), None)
    authority = _authority_policy(store, conflict, can_resolve=False, conflict=conflict)
    return (
        {
            "support_class": authority.get("support_kind") or "source_conflict_trace",
            "evidence_id": conflict.get("source_conflict_id"),
            "source_conflict_id": conflict.get("source_conflict_id"),
            "type": conflict.get("type"),
            "classification": conflict.get("classification"),
            "source_document_id": conflict.get("source_document_id"),
            "page_numbers": tuple(conflict.get("page_numbers") or conflict.get("affected_pages") or ()),
            "text_span_ids": tuple(conflict.get("text_span_ids") or ()),
            "evidence_ids": tuple(conflict.get("evidence_ids") or ()),
            "bbox_ids": tuple(conflict.get("bbox_ids") or ()),
            "bbox_count": len(conflict.get("raw_provenance_bbox_ids") or ()),
            "citation_available": False,
            "viewer_highlightable": False,
            "viewer_ref": None,
            "failure_reason": conflict.get("failure_reason") or first_reason or "source_conflict_trace_only",
        }
        | _source_conflict_contract_fields(conflict)
        | authority
    )


def _source_anomaly_answer(store, conflict: dict, query: str, *, exact_provenance: bool, trace_only: bool) -> str:
    intent = _source_conflict_intent(store)
    decision = conflict.get("resolution_decision") or {}
    folded = (query or "").casefold()
    classification = conflict.get("classification") or "source_conflict_recorded"
    summary = str(conflict.get("provenance_summary") or classification).strip()
    authority_policy = str(
        conflict.get("final_authority_policy")
        or "Sistem menampilkan provenance sumber ini sebagai jejak audit, bukan kesimpulan hukum final."
    ).strip()
    role_label = _source_conflict_role_label(store, conflict)
    reviewer_suffix = _source_conflict_reviewer_suffix(decision.get("reviewer_decision"))
    policy = conflict.get("source_anomaly_policy") or {}
    if exact_provenance:
        provenance_note = _source_conflict_provenance_note(conflict)
        return _source_anomaly_policy_answer(
            policy,
            role_label=role_label,
            summary=summary,
            authority_policy=authority_policy,
            provenance_note=provenance_note,
            reviewer_suffix=reviewer_suffix,
        )
    if trace_only:
        return _source_anomaly_policy_answer(
            policy,
            role_label=role_label,
            summary=summary,
            authority_policy=authority_policy,
            provenance_note="Jejak sumber tersedia, tetapi belum memenuhi syarat sitasi atau highlight exact.",
            reviewer_suffix=reviewer_suffix,
        )
    values = {
        "classification": classification,
        "reviewer_decision": decision.get("reviewer_decision") or "Reviewer decision unavailable",
    }
    for rule in intent.get("answer_rules") or ():
        terms = tuple(str(term).casefold() for term in rule.get("query_terms") or ())
        types = tuple(str(item) for item in rule.get("conflict_types") or ())
        if (terms and any(term in folded for term in terms)) or (types and conflict.get("type") in types):
            return str(rule.get("template") or "").format_map(values)
    return str(intent.get("default_answer_template") or "").format_map(values)


def _source_anomaly_policy_answer(
    policy: dict,
    *,
    role_label: str,
    summary: str,
    authority_policy: str,
    provenance_note: str,
    reviewer_suffix: str,
) -> str:
    values = {
        "anomaly_kind": policy.get("anomaly_kind") or "source_anomaly_provenance",
        "mapping_kind": policy.get("mapping_kind") or "source_anomaly_provenance",
        "role_label": role_label,
        "summary": summary,
        "authority_policy": authority_policy,
        "provenance_note": provenance_note,
        "reviewer_suffix": reviewer_suffix,
    }
    template = str(
        policy.get("public_wording_template")
        or "Catatan provenance sumber ({anomaly_kind}) pada {role_label}: {summary}. {authority_policy} {provenance_note}{reviewer_suffix}"
    )
    return template.format_map(values)


def _source_conflict_role_label(store, conflict: dict) -> str:
    source = _source_document_meta(store, conflict.get("source_document_id"))
    source_role = str(source.get("source_role") or "")
    labels = _source_conflict_intent(store).get("role_labels") or {}
    return str(labels.get(source_role) or source_role or conflict.get("source_document_id") or "sumber historis")


def _source_conflict_reviewer_suffix(reviewer_decision: object) -> str:
    text = str(reviewer_decision or "").strip()
    if not text:
        return ""
    return f" Reviewer decision: {text}."


def _source_conflict_contract_fields(conflict: dict) -> dict:
    raw_bbox_ids = tuple(conflict.get("raw_provenance_bbox_ids") or ())
    raw_text_span_ids = tuple(conflict.get("raw_provenance_text_span_ids") or ())
    blocked_text_span_ids = tuple(conflict.get("blocked_raw_provenance_text_span_ids") or ())
    fields = {
        "final_evidence_available": bool(conflict.get("final_evidence_available")),
        "source_anomaly_kind": conflict.get("source_anomaly_kind"),
        "source_mapping_kind": conflict.get("source_mapping_kind"),
        "provenance_bbox_status": conflict.get("provenance_bbox_status"),
        "provenance_highlight_scope": conflict.get("provenance_highlight_scope"),
        "raw_provenance_bbox_count": len(raw_bbox_ids),
        "raw_provenance_text_span_count": len(raw_text_span_ids),
        "blocked_raw_provenance_text_span_count": len(blocked_text_span_ids),
        "blocked_raw_provenance_reason": conflict.get("blocked_raw_provenance_reason"),
    }
    fields.update(_source_mapping_semantics(conflict))
    return fields


def _source_mapping_semantics(conflict: dict) -> dict:
    if conflict.get("source_anomaly_kind") != "renumbering_provenance":
        return {}
    return {
        "relation_type": "renumbered_to",
        "substantive_change": False,
        "anomaly": False,
        "source_conflict": False,
    }


def _source_conflict_provenance_note(conflict: dict) -> str:
    if conflict.get("provenance_highlight_scope") == "all_relevant_spans":
        return "Highlight viewer tersedia untuk semua span provenance exact yang relevan."
    if conflict.get("provenance_highlight_scope") == "anchor_span_only":
        return "Highlight viewer saat ini terbatas pada span anchor exact yang tersedia; span anomali lain tetap tercatat sebagai trace tanpa highlight palsu."
    return "Viewer tidak menampilkan highlight exact karena belum ada span/BBox provenance yang aman."


def _source_conflict_viewer_evidence(store, evidence_id: str | None) -> tuple[dict | None, list[dict] | None]:
    if store is None or not evidence_id:
        return None, None
    conflict = next((row for row in store.source_conflicts if row.get("source_conflict_id") == evidence_id), None)
    if conflict is None:
        return None, None
    synthetic = _synthetic_source_conflict_support(store, conflict)
    if synthetic is None:
        raw_bboxes = tuple(store.bboxes_for_refs(tuple(conflict.get("raw_provenance_bbox_ids") or ())))
        if raw_bboxes:
            synthetic = {
                "evidence": (_synthetic_source_conflict_evidence(store, conflict, raw_bboxes),),
                "bboxes": raw_bboxes,
            }
    if synthetic is None:
        return None, None
    return synthetic["evidence"][0], list(synthetic["bboxes"])


def _synthetic_source_conflict_support(store, conflict: dict) -> dict | None:
    bboxes = tuple(store.exact_bboxes_for_text_spans(tuple(conflict.get("text_span_ids") or ())))
    if not bboxes:
        return None
    evidence = _synthetic_source_conflict_evidence(store, conflict, bboxes)
    viewer_ref = {
        "action": "viewer",
        "evidence_id": evidence["evidence_id"],
        "source_document_id": evidence.get("source_document_id"),
        "page_numbers": evidence["page_numbers"],
        "bbox_count": len(bboxes),
        "can_resolve": True,
    } | _authority_policy(store, evidence, can_resolve=True, conflict=conflict)
    citation = evidence | {
        "label": conflict.get("classification"),
        "document_title": _document_title(store, _source_document_meta(store, conflict.get("source_document_id"))),
        "viewer_ref": viewer_ref,
        "bbox_count": len(bboxes),
        "evidence_status": conflict.get("status"),
    }
    return {
        "context_pack": {
            "answer_evidence": (evidence,),
            "supporting_context": (),
            "excluded_results": (),
            "citation_payloads": (),
            "viewer_refs": (viewer_ref,),
            "validation_reasons": {evidence["evidence_id"]: "source_conflict_exact_span_bbox"},
        },
        "evidence": (evidence,),
        "citations": (_citation_with_authority(store, citation, conflict=conflict),),
        "viewer_refs": (viewer_ref,),
        "bboxes": bboxes,
    }


def _synthetic_source_conflict_evidence(store, conflict: dict, bboxes: tuple[dict, ...]) -> dict:
    source = _source_document_meta(store, conflict.get("source_document_id"))
    pages = tuple(dict.fromkeys(int(row["page_number"]) for row in bboxes if row.get("page_number")))
    quoted_text = "\n".join(dict.fromkeys(str(row.get("text") or "").strip() for row in bboxes if str(row.get("text") or "").strip()))
    return {
        "evidence_id": conflict.get("source_conflict_id"),
        "legal_unit_id": None,
        "citation": f"Source anomaly: {conflict.get('classification')}",
        "quoted_text": quoted_text,
        "bbox_refs": tuple(row["bbox_id"] for row in bboxes if row.get("bbox_id")),
        "bbox_precision": "exact",
        "viewer_highlightable": True,
        "page_numbers": pages,
        "source_document_id": conflict.get("source_document_id"),
        "source_pdf_path": bboxes[0].get("source_pdf_path"),
        "source_sha256": bboxes[0].get("source_sha256"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
    }
