from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256


class FieldState(StrEnum):
    VERIFIED = "verified"
    NOT_FOUND_IN_SOURCE = "not_found_in_source"
    NOT_APPLICABLE = "not_applicable"
    NOT_YET_VERIFIED = "not_yet_verified"
    CONFLICTING_SOURCES = "conflicting_sources"
    INVALID_SOURCE_VALUE = "invalid_source_value"


class SourceKind(StrEnum):
    OFFICIAL_PDF = "official_pdf"
    OFFICIAL_CATALOG_PAGE = "official_catalog_page"
    OFFICIAL_JDIH_PAGE = "official_jdih_page"
    OFFICIAL_COURT_DECISION = "official_court_decision"
    MANUAL_REVIEW = "manual_review"


class LifecycleKind(StrEnum):
    ESTABLISHMENT = "establishment"
    SIGNING = "signing"
    PROMULGATION = "promulgation"
    EFFECTIVENESS = "effectiveness"
    EXPIRY = "expiry"


class RelationKind(StrEnum):
    AMENDS = "amends"
    AMENDED_BY = "amended_by"
    REVOKES = "revokes"
    REVOKED_BY = "revoked_by"

    @property
    def inverse(self) -> RelationKind:
        return {
            self.AMENDS: self.AMENDED_BY,
            self.AMENDED_BY: self.AMENDS,
            self.REVOKES: self.REVOKED_BY,
            self.REVOKED_BY: self.REVOKES,
        }[self]


@dataclass(frozen=True)
class SourceProvenance:
    kind: SourceKind
    reference: str
    verified_at: datetime
    immutable_source_identity: str | None = None
    page_number: int | None = None
    selector: str | None = None

    def __post_init__(self) -> None:
        if not self.reference or self.verified_at.tzinfo is None:
            raise ValueError("invalid_source_provenance")
        if self.kind is not SourceKind.OFFICIAL_PDF and (self.page_number is not None or self.selector is not None):
            raise ValueError("non_pdf_source_has_pdf_grounding")
        if (self.page_number is None) != (self.selector is None):
            raise ValueError("incomplete_pdf_grounding")


@dataclass(frozen=True)
class VerifiedValue:
    source_value: str | None
    normalized_value: str | None
    display_value: str | None
    state: FieldState
    provenance: SourceProvenance | None = None

    def __post_init__(self) -> None:
        if self.state is FieldState.VERIFIED:
            if not all((self.source_value, self.normalized_value, self.display_value, self.provenance)):
                raise ValueError("verified_value_requires_source_and_provenance")
        elif self.provenance is not None:
            raise ValueError("unverified_value_has_provenance")
        if self.state is FieldState.CONFLICTING_SOURCES and self.normalized_value is not None:
            raise ValueError("conflicting_value_collapsed")


@dataclass(frozen=True)
class LegalDocumentIdentity:
    document_type: VerifiedValue
    number: VerifiedValue
    year: VerifiedValue
    official_title: VerifiedValue
    issuer: VerifiedValue
    source_designation: VerifiedValue | None = None

    @property
    def stable_id(self) -> str:
        values = (
            self.document_type.normalized_value,
            self.number.normalized_value,
            self.year.normalized_value,
            self.issuer.normalized_value,
            self.source_designation.normalized_value if self.source_designation else "",
        )
        if any(value is None for value in values):
            raise ValueError("identity_not_verified")
        return f"legal-document-{sha256('|'.join(str(value) for value in values).encode()).hexdigest()}"


@dataclass(frozen=True)
class LifecycleEvent:
    kind: LifecycleKind
    value: VerifiedValue


@dataclass(frozen=True)
class StatusAssertion:
    status: VerifiedValue
    as_of: datetime

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("status_requires_timezone")


@dataclass(frozen=True)
class DocumentRelation:
    relation: RelationKind
    source_document_id: str
    target_document_id: str
    provenance: SourceProvenance

    def __post_init__(self) -> None:
        if self.source_document_id == self.target_document_id:
            raise ValueError("self_relation")

    def inverse(self) -> DocumentRelation:
        return DocumentRelation(self.relation.inverse, self.target_document_id, self.source_document_id, self.provenance)


@dataclass(frozen=True)
class ProvisionEffect:
    relation: RelationKind
    source_document_id: str
    target_document_id: str
    exact_target: str
    exact_source_text: str
    provenance: SourceProvenance

    def __post_init__(self) -> None:
        if self.provenance.kind is not SourceKind.OFFICIAL_PDF:
            raise ValueError("provision_effect_requires_official_pdf")
        if not self.exact_target.strip() or not self.exact_source_text.strip():
            raise ValueError("provision_effect_requires_exact_target")
        if self.provenance.selector != self.exact_source_text:
            raise ValueError("provision_effect_selector_mismatch")


@dataclass(frozen=True)
class CitationUnit:
    evidence_key: str
    document_type: str
    number: str | None
    year: str | None
    official_title: str
    publication: str | None
    provision: str | None
    page: int | None
    official_url: str
    authority: str
    citation_final: bool

    def __post_init__(self) -> None:
        if not self.evidence_key or not self.document_type or not self.official_title or not self.official_url or not self.authority:
            raise ValueError("incomplete_citation_unit")
