from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
import re
import unicodedata

from tjipto.contracts.legal_information import (
    CitationUnit,
    DocumentRelation,
    FieldState,
    LegalDocumentIdentity,
    LifecycleEvent,
    ProvisionEffect,
    StatusAssertion,
    VerifiedValue,
)
from tjipto.contracts.source_text import SourceAnnotation
CATALOG_FILTERS = ("document_role", "legal_status", "establishment_period")


def normalize_catalog_text(value: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", unicodedata.normalize("NFKC", value).casefold()).split())


@dataclass(frozen=True)
class CatalogDocument:
    identity: LegalDocumentIdentity
    short_title: str
    aliases: tuple[str, ...]
    legal_status: StatusAssertion
    document_role: str
    document_role_label: str
    lifecycle: tuple[LifecycleEvent, ...]
    relations: tuple[DocumentRelation, ...]
    provision_effects: tuple[ProvisionEffect, ...]
    publication: VerifiedValue
    official_url: str
    source_path: Path
    source_sha256: str
    page_count: int
    preferred: bool
    permissions: frozenset[str]
    corpus_id: str | None = None
    source_annotations: tuple[SourceAnnotation, ...] = ()

    @property
    def stable_id(self) -> str:
        return self.identity.stable_id

    @property
    def public_target_id(self) -> str:
        return sha256(f"catalog-target|{self.stable_id}".encode()).hexdigest()

    @property
    def citation_unit(self) -> CitationUnit:
        return CitationUnit(
            evidence_key=self.stable_id,
            document_type=self.identity.document_type.display_value or "",
            number=self.identity.number.display_value,
            year=self.identity.year.display_value,
            official_title=self.identity.official_title.display_value or "",
            publication=self.publication.display_value if self.publication.state is FieldState.VERIFIED else None,
            provision=None,
            page=None,
            official_url=self.official_url,
            authority="official_catalog_page",
            citation_final=False,
        )

    @property
    def establishment_period(self) -> str:
        establishment = next(
            (
                event.value
                for event in self.lifecycle
                if event.kind.value == "establishment" and event.value.state is FieldState.VERIFIED
            ),
            None,
        )
        match = re.match(r"(\d{4})", establishment.normalized_value or "") if establishment else None
        if match is None:
            return ""
        year = int(match.group(1))
        return f"{year // 10 * 10}-{year // 10 * 10 + 9}"

    def __post_init__(self) -> None:
        if not self.permissions <= {"catalog", "view"}:
            raise ValueError("catalog_permission_escalation")
        if "catalog" not in self.permissions or "view" not in self.permissions:
            raise ValueError("catalog_permission_missing")
        for relation in self.relations:
            if relation.source_document_id != self.stable_id and relation.target_document_id != self.stable_id:
                raise ValueError("relation_endpoint_mismatch")
        for effect in self.provision_effects:
            if not {effect.source_document_id, effect.target_document_id} <= {
                self.stable_id,
                *(relation.source_document_id for relation in self.relations),
                *(relation.target_document_id for relation in self.relations),
            }:
                raise ValueError("effect_endpoint_mismatch")


@dataclass(frozen=True)
class CatalogQuery:
    text: str
    limit: int = 10
    filters: tuple[tuple[str, str], ...] = ()
    corpus_id: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= 50:
            raise ValueError("invalid_limit")
        if any(name not in CATALOG_FILTERS or not value for name, value in self.filters):
            raise ValueError("invalid_catalog_filter")


class CatalogRepository:
    def __init__(self, documents: tuple[CatalogDocument, ...]):
        stable_ids = [document.stable_id for document in documents]
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("duplicate_catalog_document")
        relations = {relation for document in documents for relation in document.relations}
        descriptors = {(relation.relation, relation.source_document_id, relation.target_document_id) for relation in relations}
        if any(
            (relation.relation.inverse, relation.target_document_id, relation.source_document_id) not in descriptors
            for relation in relations
        ):
            raise ValueError("missing_inverse_relation")
        self.documents = documents
        self._by_target = {document.public_target_id: document for document in documents}

    def resolve(self, public_target_id: str) -> CatalogDocument | None:
        return self._by_target.get(public_target_id)

    def search(self, query: CatalogQuery) -> tuple[tuple[CatalogDocument, ...], tuple[dict, ...], int]:
        normalized = normalize_catalog_text(query.text)
        filters = dict(query.filters)
        candidates = tuple(
            document
            for document in self.documents
            if (query.corpus_id is None or document.corpus_id == query.corpus_id)
            and self._matches_filters(document, filters)
        )
        ranked = sorted(
            ((self._score(document, normalized), document) for document in candidates),
            key=lambda item: (item[0], item[1].stable_id),
            reverse=True,
        )
        matched = tuple(document for score, document in ranked if score[0] > 0)
        facets = self.facets(normalized, corpus_id=query.corpus_id)
        return matched[: query.limit], facets, len(matched)

    def facets(self, normalized_query: str = "", *, corpus_id: str | None = None) -> tuple[dict, ...]:
        documents = tuple(
            document
            for document in self.documents
            if (corpus_id is None or document.corpus_id == corpus_id)
            and (not normalized_query or self._score(document, normalized_query)[0] > 0)
        )
        values = {
            "document_role": ((document.document_role, document.document_role_label) for document in documents),
            "legal_status": (
                (document.legal_status.status.normalized_value or "", document.legal_status.status.display_value or "")
                for document in documents
            ),
            "establishment_period": ((document.establishment_period, document.establishment_period) for document in documents),
        }
        result = []
        labels = {
            "document_role": "Kedudukan Naskah",
            "legal_status": "Status Keberlakuan",
            "establishment_period": "Periode Penetapan",
        }
        for name in CATALOG_FILTERS:
            counts: dict[tuple[str, str], int] = {}
            for value, label in values[name]:
                if value:
                    counts[(value, label)] = counts.get((value, label), 0) + 1
            result.append(
                {
                    "name": name,
                    "label": labels[name],
                    "options": tuple(
                        {"value": value, "label": label, "count": count}
                        for (value, label), count in sorted(counts.items())
                    ),
                }
            )
        return tuple(result)

    @staticmethod
    def _matches_filters(document: CatalogDocument, filters: dict[str, str]) -> bool:
        values = {
            "document_role": document.document_role,
            "legal_status": document.legal_status.status.normalized_value,
            "establishment_period": document.establishment_period,
        }
        return all(values[name] == value for name, value in filters.items())

    @staticmethod
    def _score(document: CatalogDocument, query: str) -> tuple[int, int, int, float]:
        if not query:
            return (1, int(document.preferred), int(document.identity.year.normalized_value or 0), 0.0)
        identity = normalize_catalog_text(
            " ".join(
                (
                    document.identity.document_type.normalized_value or "",
                    document.identity.number.normalized_value or "",
                    document.identity.year.normalized_value or "",
                    document.identity.official_title.normalized_value or "",
                )
            )
        )
        aliases = tuple(normalize_catalog_text(alias) for alias in document.aliases)
        title = normalize_catalog_text(
            " ".join((document.short_title, document.identity.official_title.normalized_value or "", *document.aliases))
        )
        explicit_history = any(term in query.split() for term in ("asli", "historis", "perubahan", "amandemen"))
        exact_identity = query == identity
        exact_alias = query in aliases
        structured = all(token in identity.split() for token in query.split()) and any(token.isdigit() for token in query.split())
        query_tokens = tuple(token for token in query.split() if token not in {"tahun", "nomor", "tentang"})
        lexical = sum(1 for token in query_tokens if token in title.split())
        lexical_match = lexical >= (1 if len(query_tokens) == 1 else max(2, (len(query_tokens) + 1) // 2))
        typo = SequenceMatcher(None, query, title).ratio() if len(query) >= 5 else 0.0
        relevance = 6 if exact_identity else 5 if exact_alias else 4 if structured else 3 if lexical_match else 2 if typo >= 0.82 else 0
        preferred = int(document.preferred and not explicit_history)
        return (relevance, preferred, int(document.identity.year.normalized_value or 0), typo)


class CatalogService:
    def __init__(self, repository: CatalogRepository):
        self.repository = repository

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        *,
        corpus_id: str | None = None,
    ) -> dict:
        try:
            typed_filters = tuple(sorted((str(name), str(value)) for name, value in (filters or {}).items()))
            documents, facets, total = self.repository.search(CatalogQuery(query, limit, typed_filters, corpus_id))
        except ValueError:
            return {"status": "invalid_filter", "results": (), "facets": self.repository.facets(), "total": 0}
        return {
            "status": "found" if documents else "no_results",
            "results": documents,
            "facets": facets,
            "total": total,
            "applied_filters": dict(typed_filters),
        }

    def document(self, target: str) -> CatalogDocument | None:
        return self.repository.resolve(target)

    def pdf(self, target: str) -> dict:
        document = self.repository.resolve(target)
        return {"status": "pdf_access_ready", "path": document.source_path} if document else {"status": "not_found"}
