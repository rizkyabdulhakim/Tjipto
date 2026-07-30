from __future__ import annotations

from dataclasses import dataclass, field

from tjipto.contracts.legal_information import CitationUnit


@dataclass(frozen=True)
class IndonesianLegalCitationProfile:
    """Deterministic citation rendering from typed legal identity only."""

    name: str = "Sitasi Hukum Indonesia Tjipto"

    def full(self, unit: CitationUnit) -> str:
        identity = " ".join(part for part in (unit.document_type, unit.number, unit.year) if part)
        parts = [f"{identity} tentang {unit.official_title}"]
        if unit.provision:
            parts.append(unit.provision)
        if unit.publication:
            parts.append(unit.publication)
        if unit.page is not None:
            parts.append(f"hlm. {unit.page}")
        return ", ".join(parts)

    def short(self, unit: CitationUnit) -> str:
        identity = " ".join(part for part in (unit.document_type, unit.number, unit.year) if part)
        return ", ".join(part for part in (identity, unit.provision) if part)


@dataclass
class FootnoteBook:
    profile: IndonesianLegalCitationProfile = field(default_factory=IndonesianLegalCitationProfile)
    _numbers: dict[str, int] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)

    def cite(self, unit: CitationUnit) -> tuple[int, str]:
        number = self._numbers.setdefault(unit.evidence_key, len(self._numbers) + 1)
        subsequent = unit.evidence_key in self._seen
        self._seen.add(unit.evidence_key)
        return number, self.profile.short(unit) if subsequent else self.profile.full(unit)

    def footnote(self, units: tuple[CitationUnit, ...]) -> tuple[int, str]:
        if not units:
            raise ValueError("empty_footnote")
        entries = [self.cite(unit) for unit in units]
        number = min(item[0] for item in entries)
        return number, "; ".join(text for _, text in entries)
