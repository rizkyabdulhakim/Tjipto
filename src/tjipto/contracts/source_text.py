from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceTextDisposition(StrEnum):
    LEGAL_TEXT = "legal_text"
    STRUCTURAL_TEXT = "structural_text"
    INSTRUMENT_TEXT = "instrument_text"
    SOURCE_FACT = "source_fact"
    SOURCE_CONFLICT = "source_conflict"
    SOURCE_ANNOTATION = "source_annotation"
    DOCUMENT_FURNITURE = "document_furniture"
    LAYOUT_SEPARATOR = "layout_separator"
    EXTRACTION_ARTIFACT = "extraction_artifact"


class SourceTextCapability(StrEnum):
    LEGAL_ANSWER = "legal_answer"
    STRUCTURAL_ANSWER = "structural_answer"
    INSTRUMENT_ANSWER = "instrument_answer"
    SOURCE_FACT_ANSWER = "source_fact_answer"
    DISCREPANCY_ANSWER = "discrepancy_answer"
    ANNOTATION_ANSWER = "annotation_answer"
    SOURCE_FORMAT_ANSWER = "source_format_answer"
    AUDIT_ONLY = "audit_only"


@dataclass(frozen=True)
class SourceSelector:
    stream_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.stream_id or self.start < 0 or self.end <= self.start:
            raise ValueError("invalid_source_selector")


@dataclass(frozen=True)
class SourceTextRecord:
    source_value: str
    normalized_value: str
    source_document_id: str
    source_sha256: str
    page_number: int
    extraction_order: int
    selector: SourceSelector
    geometry_available: bool
    disposition: SourceTextDisposition
    legal_force: str
    capabilities: tuple[SourceTextCapability, ...]
    legal_answer_eligible: bool
    source_answer_eligible: bool
    legal_citation_eligible: bool
    source_citation_eligible: bool
    default_highlight_eligible: bool
    abstention_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.source_value or not self.normalized_value:
            raise ValueError("empty_source_text")
        if not self.source_document_id or not self.source_sha256 or self.page_number < 1 or self.extraction_order < 0:
            raise ValueError("invalid_source_identity")
        if not self.capabilities and not self.abstention_reason:
            raise ValueError("source_text_without_route_or_review")
        if self.disposition is SourceTextDisposition.SOURCE_ANNOTATION and (
            self.legal_answer_eligible or self.legal_citation_eligible or self.default_highlight_eligible
        ):
            raise ValueError("source_annotation_promoted_to_law")


@dataclass(frozen=True)
class SourceAnnotation:
    marker: str
    meaning: str
    source_document_id: str
    source_sha256: str
    page_number: int
    selector: SourceSelector
    legend_support_id: str
    related_source_roles: tuple[str, ...]
    target_legal_unit_id: str | None = None
    source_citation_eligible: bool = True
    legal_citation_eligible: bool = False
    default_highlight_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.marker or not self.meaning or not self.legend_support_id:
            raise ValueError("incomplete_source_annotation")
        if self.legal_citation_eligible or self.default_highlight_eligible:
            raise ValueError("source_annotation_promoted_to_law")


@dataclass(frozen=True)
class SourceTextQueryResult:
    route: str
    answer: str
    annotations: tuple[SourceAnnotation, ...]
    supports: tuple[dict, ...]
    target_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.route or not self.answer or not self.annotations or not self.supports:
            raise ValueError("incomplete_source_text_result")
