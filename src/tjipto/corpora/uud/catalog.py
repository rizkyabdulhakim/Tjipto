from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import re

from tjipto.catalog import CatalogDocument
from tjipto.contracts.legal_information import (
    FieldState,
    DocumentRelation,
    LegalDocumentIdentity,
    LifecycleEvent,
    LifecycleKind,
    ProvisionEffect,
    RelationKind,
    SourceKind,
    SourceProvenance,
    StatusAssertion,
    VerifiedValue,
)
from tjipto.corpora.uud.source_annotations import document_annotations


_TITLES = {
    "current_consolidated": ("Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 dalam Satu Naskah", "UUD 1945 Satu Naskah", True, "consolidated", "Naskah Konsolidasi"),
    "original_historical": ("Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "UUD 1945 Naskah Asli", False, "original", "Naskah Asli"),
    "amendment_1_historical": ("Perubahan Pertama Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Pertama UUD 1945", False, "constitutional_amendment", "Amandemen"),
    "amendment_2_historical": ("Perubahan Kedua Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Kedua UUD 1945", False, "constitutional_amendment", "Amandemen"),
    "amendment_3_historical": ("Perubahan Ketiga Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Ketiga UUD 1945", False, "constitutional_amendment", "Amandemen"),
    "amendment_4_historical": ("Perubahan Keempat Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Keempat UUD 1945", False, "constitutional_amendment", "Amandemen"),
}
_YEARS = {
    "current_consolidated": "1945",
    "original_historical": "1945",
    "amendment_1_historical": "1999",
    "amendment_2_historical": "2000",
    "amendment_3_historical": "2001",
    "amendment_4_historical": "2002",
}


def citation_identity(source_role: str) -> tuple[str, str, str]:
    if source_role not in _TITLES:
        raise ValueError("unknown_uud_source_role")
    return "Undang-Undang Dasar", _YEARS[source_role], _TITLES[source_role][0]


def documents(store) -> tuple[CatalogDocument, ...]:
    verified_at = datetime(2026, 7, 30, tzinfo=timezone.utc)
    sources = {str(source.get("source_role")): source for source in store.source_documents if source.get("source_role") in _TITLES}
    artifacts = store.config.json("runtime_projection").get("artifacts", {})
    graph_projections = tuple(
        row["relation_projection"]
        for row in store.graph_edges
        if isinstance(row.get("relation_projection"), dict)
    )
    metadata = _metadata_by_role(tuple(artifacts.get("metadata_grounding") or ()))
    document_metadata = _document_metadata_by_role(tuple(artifacts.get("document_metadata") or ()))
    evidence_by_id = {
        str(row["evidence_id"]): row
        for row in artifacts.get("evidence_registry") or ()
        if row.get("evidence_id")
    }
    identities = {
        role: _identity(role, source, metadata.get(role, {}).get("institution"), verified_at)
        for role, source in sources.items()
    }
    stable_ids = {role: identity.stable_id for role, identity in identities.items()}
    result = []
    for role, source in sources.items():
        role = str(source.get("source_role"))
        title, short_title, preferred, document_role, role_label = _TITLES[role]
        identity = identities[role]
        relations = _document_relations(
            role,
            stable_ids,
            sources,
            graph_projections,
            verified_at,
        )
        effects = _provision_effects(
            role,
            stable_ids,
            source,
            graph_projections,
            evidence_by_id,
            verified_at,
        )
        establishment = metadata.get(role, {}).get("penetapan")
        place = metadata.get(role, {}).get("place")
        publication = metadata.get(role, {}).get("source_publication")
        signatories = _signatory_value(
            metadata.get(role, {}).get("signatories"),
            document_metadata.get(role),
            source,
            verified_at,
        )
        aliases = [short_title, title, f"{role_label} UUD 1945", "konstitusi indonesia"]
        if role == "current_consolidated":
            aliases.append("Undang-Undang Dasar Negara Republik Indonesia Tahun 1945")
        result.append(
            CatalogDocument(
                identity,
                short_title,
                tuple(aliases),
                _status_assertion(source, verified_at),
                document_role,
                role_label,
                (_establishment_event(establishment, source, verified_at),) if establishment else (),
                relations,
                effects,
                _publication_value(publication, source, verified_at) if publication else VerifiedValue(None, None, None, FieldState.NOT_APPLICABLE),
                str(source["download_url"]),
                store.config.source_path(str(source["path"])),
                str(source["sha256"]),
                int(source["page_count"]),
                preferred,
                frozenset({"catalog", "view"}),
                corpus_id="uud",
                source_annotations=document_annotations(store) if role == "current_consolidated" else (),
                establishment_place=_place_value(place, source, verified_at) if place else None,
                signatories=signatories,
            )
        )
    return tuple(result)


def _identity(role: str, source: dict, institution: dict | None, verified_at: datetime) -> LegalDocumentIdentity:
    title, short_title, *_ = _TITLES[role]
    provenance = _source_provenance(source, verified_at)

    def value(source_value: str, normalized: str | None = None) -> VerifiedValue:
        return VerifiedValue(source_value, normalized or source_value.casefold(), source_value, FieldState.VERIFIED, provenance)

    issuer = (
        _institution_value(role, institution, source, verified_at)
        if institution
        else VerifiedValue(None, None, None, FieldState.NOT_FOUND_IN_SOURCE)
    )
    return LegalDocumentIdentity(
        value("Undang-Undang Dasar", "undang-undang dasar"),
        VerifiedValue(None, None, None, FieldState.NOT_APPLICABLE),
        value(_YEARS[role], _YEARS[role]),
        value(title, title.casefold()),
        issuer,
        VerifiedValue(str(source["filename"]), role, short_title, FieldState.VERIFIED, provenance),
    )


def _status_assertion(source: dict, verified_at: datetime) -> StatusAssertion:
    reference = str(source["source_page_url"])
    provenance = SourceProvenance(
        SourceKind.OFFICIAL_CATALOG_PAGE,
        reference,
        verified_at,
        f"{reference}|{source['sha256']}",
        source_authority=str(source.get("source_authority") or "BPK Database Peraturan"),
    )
    return StatusAssertion(
        VerifiedValue("Berlaku", "berlaku", "Berlaku", FieldState.VERIFIED, provenance),
        verified_at,
        "parent_record",
    )


def _source_provenance(source: dict, verified_at: datetime, *, page: int | None = None, selector: str | None = None) -> SourceProvenance:
    return SourceProvenance(
        SourceKind.OFFICIAL_PDF,
        str(source["download_url"]),
        verified_at,
        str(source["sha256"]),
        page,
        selector,
    )


def _metadata_by_role(rows: tuple[dict, ...]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        role, field = row.get("source_role"), row.get("metadata_field")
        if role in _TITLES and field in {"institution", "penetapan", "place", "signatories", "source_publication"} and row.get("quoted_text"):
            current = result[str(role)].get(str(field))
            if current is None or len(str(row["quoted_text"])) > len(str(current["quoted_text"])):
                result[str(role)][str(field)] = row
    return dict(result)


def _document_metadata_by_role(rows: tuple[dict, ...]) -> dict[str, dict]:
    return {
        str(row["source_role"]): row
        for row in rows
        if row.get("source_role") in _TITLES and isinstance(row, dict)
    }


def _metadata_value(row: dict, source: dict, verified_at: datetime, display: str, normalized: str) -> VerifiedValue:
    quote = str(row["quoted_text"])
    page = next(iter(row.get("page_numbers") or ()), None)
    return VerifiedValue(quote, normalized, display, FieldState.VERIFIED, _source_provenance(source, verified_at, page=page, selector=quote))


def _institution_value(role: str, row: dict, source: dict, verified_at: datetime) -> VerifiedValue:
    display = (
        "Sekretariat Jenderal Majelis Permusyawaratan Rakyat Republik Indonesia"
        if role == "current_consolidated"
        else "Majelis Permusyawaratan Rakyat Republik Indonesia"
    )
    return _metadata_value(row, source, verified_at, display, display.casefold())


def _publication_value(row: dict, source: dict, verified_at: datetime) -> VerifiedValue:
    display = "Sekretariat Jenderal MPR RI — Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 dalam Satu Naskah"
    return _metadata_value(row, source, verified_at, display, display.casefold())


def _establishment_event(row: dict, source: dict, verified_at: datetime) -> LifecycleEvent:
    quote = " ".join(str(row["quoted_text"]).split())
    display = re.sub(r"^Ditetapkan di\s+\S+\s+Pada tanggal\s+", "", quote, flags=re.IGNORECASE)
    return LifecycleEvent(LifecycleKind.ESTABLISHMENT, _metadata_value(row, source, verified_at, display, _normalized_date(display)))


def _place_value(row: dict, source: dict, verified_at: datetime) -> VerifiedValue:
    display = re.sub(r"^Ditetapkan di\s+", "", " ".join(str(row["quoted_text"]).split()), flags=re.IGNORECASE)
    return _metadata_value(row, source, verified_at, display, display.casefold())


def _signatory_value(
    row: dict | None,
    document_metadata: dict | None,
    source: dict,
    verified_at: datetime,
) -> VerifiedValue | None:
    if row is None or not document_metadata:
        return None
    names = tuple(
        str(item.get("name_text", "")).strip()
        for item in document_metadata.get("signatories", ())
        if isinstance(item, dict) and str(item.get("name_text", "")).strip()
    )
    if not names:
        return None
    display = "; ".join(dict.fromkeys(names))
    return _metadata_value(row, source, verified_at, display, display.casefold())


def _normalized_date(value: str) -> str:
    months = {
        "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
        "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
    }
    match = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", value)
    if not match or match.group(2).casefold() not in months:
        return value.casefold()
    return f"{int(match.group(3)):04d}-{months[match.group(2).casefold()]:02d}-{int(match.group(1)):02d}"


def _document_relations(
    role: str,
    stable_ids: dict[str, str],
    sources: dict[str, dict],
    rows: tuple[dict, ...],
    verified_at: datetime,
) -> tuple[DocumentRelation, ...]:
    relation_types = {
        "AMENDS": RelationKind.AMENDS,
        "AMENDED_BY": RelationKind.AMENDED_BY,
        "CONSOLIDATES": RelationKind.CONSOLIDATES,
        "CONSOLIDATED_BY": RelationKind.CONSOLIDATED_BY,
        "DERIVED_FROM": RelationKind.DERIVED_FROM,
        "DERIVES": RelationKind.DERIVES,
    }
    amendment_roles = tuple(f"amendment_{index}_historical" for index in range(1, 5))
    panel_rows = _panel_amendment_relation_rows(role, amendment_roles)
    graph_rows = (
        row
        for row in rows
        if str(row.get("source_role")) == role
        and (
            role not in {"original_historical", *amendment_roles}
            or str(row.get("relation_type")) not in {"AMENDS", "AMENDED_BY"}
        )
    )
    candidate_rows = (*graph_rows, *panel_rows)
    seen: set[tuple[str, str, str]] = set()
    result = []
    for row in candidate_rows:
        source_role, target_role = str(row.get("source_role")), str(row.get("target_source_role"))
        relation_type = str(row.get("relation_type"))
        key = (relation_type, source_role, target_role)
        if (
            source_role != role
            or relation_type not in relation_types
            or source_role not in stable_ids
            or target_role not in stable_ids
            or key in seen
        ):
            continue
        seen.add(key)
        proof_role = target_role if relation_type in {"AMENDED_BY", "CONSOLIDATED_BY", "DERIVES"} else source_role
        result.append(
            DocumentRelation(
                relation_types[relation_type],
                stable_ids[source_role],
                stable_ids[target_role],
                _source_provenance(sources[proof_role], verified_at),
            )
        )
    return tuple(result)


def _panel_amendment_relation_rows(role: str, amendment_roles: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    """Project chronological amendment navigation without changing legal graph edges."""
    sequence = ("original_historical", *amendment_roles)
    if role not in sequence:
        return ()
    position = sequence.index(role)
    result = []
    if position:
        result.append({"relation_type": "AMENDS", "source_role": role, "target_source_role": sequence[position - 1]})
    if position < len(sequence) - 1:
        result.append({"relation_type": "AMENDED_BY", "source_role": role, "target_source_role": sequence[position + 1]})
    return tuple(result)


def _provision_effects(
    role: str,
    stable_ids: dict[str, str],
    source: dict,
    rows: tuple[dict, ...],
    evidence_by_id: dict[str, dict],
    verified_at: datetime,
) -> tuple[ProvisionEffect, ...]:
    effect_types = {"MODIFIES", "RENAMES", "RENUMBERED_TO", "DELETES", "ADDS", "AMBIGUOUS_OPERATION"}
    effects = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        relation_type = str(row.get("relation_type"))
        evidence = evidence_by_id.get(str(row.get("evidence_id")), {})
        quote = str(row.get("quoted_text") or evidence.get("quoted_text") or "")
        if (
            str(row.get("support_source_role") or row.get("source_role")) == role
            and row.get("runtime_loadable") is True
            and row.get("source_support_exact") is True
            and relation_type in effect_types
            and row.get("evidence_id")
            and quote
        ):
            exact_target = _public_target_label(row.get("target_citation") or row.get("target_label"))
            key = (relation_type, exact_target, quote)
            if not exact_target or key in seen:
                continue
            seen.add(key)
            # Provision effects are owned by the amendment document.  The
            # article relation keeps the historical predecessor unit, while
            # this catalog-level effect remains attached to its owner so the
            # public document contract does not require an article edge.
            target_role = role
            effects.append(
                ProvisionEffect(
                    RelationKind.AMENDS,
                    stable_ids[role],
                    stable_ids.get(target_role, stable_ids[role]),
                    exact_target,
                    quote,
                    _source_provenance(source, verified_at, page=row.get("page_number"), selector=quote),
                    operation=relation_type,
                )
            )
    return tuple(effects)


def _public_target_label(value: object) -> str:
    return re.sub(r"\s+(?:scope|clause\s*\([^)]*\))$", "", str(value or "").strip(), flags=re.IGNORECASE)
