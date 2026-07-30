from __future__ import annotations

from dataclasses import dataclass, field

from tjipto.contracts.legal_information import CitationUnit


@dataclass(frozen=True)
class IndonesianLegalCitationProfile:
    """Deterministic citation rendering from typed legal identity only."""

    name: str = "Sitasi Hukum Indonesia Tjipto"

    def full(self, unit: CitationUnit) -> str:
        identity = _legal_identity(unit)
        parts = [identity if identity.casefold() == unit.official_title.casefold() else f"{identity} tentang {unit.official_title}"]
        if unit.provision:
            parts.append(unit.provision)
        if unit.publication:
            parts.append(unit.publication)
        if unit.page is not None:
            parts.append(f"hlm. {unit.page}")
        return ", ".join(parts)

    def short(self, unit: CitationUnit) -> str:
        return ", ".join(part for part in (_short_identity(unit), unit.provision) if part)


@dataclass
class FootnoteBook:
    profile: IndonesianLegalCitationProfile = field(default_factory=IndonesianLegalCitationProfile)
    _next_number: int = 1
    _seen_documents: set[tuple[str | None, ...]] = field(default_factory=set)

    def cite(self, unit: CitationUnit) -> tuple[int, str]:
        return self.footnote((unit,))

    def footnote(self, units: tuple[CitationUnit, ...]) -> tuple[int, str]:
        if not units:
            raise ValueError("empty_footnote")
        ordered = tuple(sorted(set(units), key=_citation_order))
        number = self._next_number
        self._next_number += 1
        entries = []
        for unit in ordered:
            document = _document_identity(unit)
            entries.append(self.profile.short(unit) if document in self._seen_documents else self.profile.full(unit))
            self._seen_documents.add(document)
        return number, "; ".join(entries)


def _document_identity(unit: CitationUnit) -> tuple[str | None, ...]:
    return (
        unit.document_type,
        unit.number,
        unit.year,
        unit.official_title,
    )


def _citation_order(unit: CitationUnit) -> tuple[str, ...]:
    return tuple(
        "" if value is None else str(value)
        for value in (
            unit.document_type,
            unit.number,
            unit.year,
            unit.official_title,
            unit.provision,
            unit.page,
            unit.official_url,
            unit.authority,
            unit.citation_final,
            unit.evidence_key,
        )
    )


def _legal_identity(unit: CitationUnit) -> str:
    if unit.number:
        return " ".join(
            part for part in (unit.document_type, f"Nomor {unit.number}", f"Tahun {unit.year}" if unit.year else None) if part
        )
    if unit.official_title.casefold().startswith(unit.document_type.casefold()):
        return unit.official_title
    return " ".join(part for part in (unit.document_type, f"Tahun {unit.year}" if unit.year else None) if part)


def _short_identity(unit: CitationUnit) -> str:
    if unit.number:
        return " ".join(
            part for part in (unit.document_type, f"Nomor {unit.number}", f"Tahun {unit.year}" if unit.year else None) if part
        )
    return " ".join(part for part in (unit.document_type, f"Tahun {unit.year}" if unit.year else None) if part)
