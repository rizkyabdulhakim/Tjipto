from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from collections import OrderedDict
from typing import Any
from uuid import uuid4

from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.strategy import StrategyRegistry
from tjipto.corpora.verified import VerifiedCorpusRepository
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.answer import assemble_context_pack
from tjipto.retrieval.metadata import (
    normalize_filters,
    public_filters,
)
from tjipto.retrieval.research import (
    ResearchIntent,
    ResearchPlanningProvider,
    TaskPlan,
    research_planning_provider_from_environment,
)
from tjipto.retrieval.sufficiency import EvidenceRequirement
from tjipto.runtime.answer_arbitration import (
    _empty_citation_fields,
)
from tjipto.runtime.bookmarks import BookmarkRepository
from tjipto.runtime.public_document import (
    _catalog_search,
    _catalog_search_response,
)
from tjipto.runtime.query_semantics import interpret_query
from tjipto.runtime.orchestration import OrchestrationController
from tjipto.runtime.response import _integrity_failure
from tjipto.runtime.wording import wording_provider_from_environment
from tjipto.runtime.scope_guard import scope_guard_context
from tjipto.telemetry import Telemetry
from tjipto.runtime.viewer import (
    _citation_with_authority,
    _source_status_label,
    pdf_access_request,
    viewer_request,
)
from tjipto.catalog import CatalogService


_PROVIDER_FROM_ENVIRONMENT = object()


class LegalRuntimeService:
    def __init__(
        self,
        repo_root: Path | None = None,
        telemetry: Telemetry | None = None,
        strategy_registry: StrategyRegistry | None = None,
        *,
        answer_provider: Any = _PROVIDER_FROM_ENVIRONMENT,
        planning_provider: Any = _PROVIDER_FROM_ENVIRONMENT,
    ):
        self.registry = CorpusRegistry(repo_root, strategies=strategy_registry)
        self.repository = VerifiedCorpusRepository(self.registry)
        resolved_telemetry = telemetry or Telemetry.from_environment(self.registry)
        resolved_telemetry.bind_registry(self.registry)
        self._public_targets: OrderedDict[str, tuple[str, dict]] = OrderedDict()
        self._public_target_limit = 1024
        self._public_target_lock = RLock()
        self._catalog_service = None
        self._bookmarks = BookmarkRepository()
        self._answer_provider = wording_provider_from_environment() if answer_provider is _PROVIDER_FROM_ENVIRONMENT else answer_provider
        resolved_planning_provider = (
            research_planning_provider_from_environment() if planning_provider is _PROVIDER_FROM_ENVIRONMENT else planning_provider
        )
        self._orchestrator = OrchestrationController(
            self.registry,
            self.repository,
            resolved_telemetry,
            answer_provider=self._answer_provider,
            planning_provider=resolved_planning_provider,
        )

    def _store(self, corpus_id: str):
        return self._orchestrator.store(corpus_id)

    @property
    def _integrity_error(self) -> str | None:
        return self._orchestrator.integrity_error

    @property
    def telemetry(self) -> Telemetry:
        return self._orchestrator.telemetry

    @telemetry.setter
    def telemetry(self, telemetry: Telemetry) -> None:
        telemetry.bind_registry(self.registry)
        self._orchestrator.telemetry = telemetry

    @property
    def _planning_provider(self):
        return self._orchestrator.planning_provider

    @_planning_provider.setter
    def _planning_provider(self, provider) -> None:
        self._orchestrator.planning_provider = provider

    def _route_retrieval(self, corpus_id: str, query: str, store: EvidenceStore, **kwargs: Any) -> dict:
        return self._orchestrator.route_retrieval(corpus_id, query, store, **kwargs)

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
        required_entities: tuple[str, ...] = (),
        explicit_references: tuple[str, ...] = (),
        source_role: str | None = None,
        temporal_scope: str | None = None,
        polarity: str | None = None,
        modality: str | None = None,
        plan: TaskPlan | None = None,
    ) -> dict:
        return self._orchestrator.research(
            corpus_id,
            query,
            intent=intent,
            requirements=requirements,
            planning_provider=planning_provider,
            limit=limit,
            max_rounds=max_rounds,
            required_entities=required_entities,
            explicit_references=explicit_references,
            source_role=source_role,
            temporal_scope=temporal_scope,
            polarity=polarity,
            modality=modality,
            plan=plan,
        )

    def _telemetry_corpus_id(self, corpus_id: str) -> str:
        return self._orchestrator.telemetry_corpus_id(corpus_id)

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
                if document.identity.source_designation is not None and document.identity.source_designation.normalized_value == role
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
        semantics = interpret_query(store, corpus_id, query, available_corpora=self.registry.corpus_ids())
        requested_role = source_role or (semantics.source_scopes[0] if semantics.source_scopes else None)
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
        return viewer_request(
            store,
            corpus_id,
            evidence_id,
            support_unit_id=support_unit_id,
            source_support_id=source_support_id,
            relation_id=relation_id,
            source_document_id=source_document_id,
            page_number=page_number,
            bbox_id=bbox_id,
            bbox_refs=bbox_refs,
            proposition_id=proposition_id,
            quoted_text=quoted_text,
            support_projection=support_projection,
            source_pdf_path=source_pdf_path,
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
        return pdf_access_request(
            store,
            corpus_id,
            evidence_id,
            source_support_id=source_support_id,
            relation_id=relation_id,
            source_document_id=source_document_id,
            page_number=page_number,
            source_sha256=source_sha256,
            bbox_id=bbox_id,
            bbox_refs=bbox_refs,
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
        evidence_requirements: tuple[EvidenceRequirement, ...] = (),
        clarification_id: str | None = None,
        clarification_answer: str | None = None,
    ) -> dict:
        return self._orchestrator.ask(
            corpus_id,
            query,
            limit,
            filters,
            evidence_requirements,
            clarification_id,
            clarification_answer,
        )

    def _ask(
        self,
        corpus_id: str,
        query: str,
        limit: int = 3,
        filters: dict | None = None,
        evidence_requirements: tuple[EvidenceRequirement, ...] = (),
        clarification_id: str | None = None,
        clarification_answer: str | None = None,
        summary_mode: bool = False,
    ) -> dict:
        return self._orchestrator.ask_raw(
            corpus_id,
            query,
            limit,
            filters,
            evidence_requirements,
            clarification_id,
            clarification_answer,
            summary_mode,
        )

    def _clarification_response(self, corpus_id: str, query: str, plan: TaskPlan, round_number: int) -> dict:
        return self._orchestrator.clarification_response(corpus_id, query, plan, round_number)

    def _resume_clarification(
        self,
        corpus_id: str,
        clarification_id: str | None,
        clarification_answer: str | None,
    ) -> tuple[str, int] | dict | None:
        return self._orchestrator.resume_clarification(corpus_id, clarification_id, clarification_answer)
