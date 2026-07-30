from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re

import pymupdf

from tjipto.catalog import CatalogDocument
from tjipto.contracts.legal_information import (
    DocumentRelation,
    FieldState,
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


def documents(repo_root: Path) -> tuple[CatalogDocument, ...]:
    path = repo_root / "data" / "catalog" / "regulations.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    identities = {record["key"]: _identity(record) for record in records["documents"]}
    stable_ids = {key: identity.stable_id for key, identity in identities.items()}
    result = []
    for record in records["documents"]:
        identity = identities[record["key"]]
        source_path = (repo_root / record["acquisition"]["path"]).resolve()
        if not source_path.is_relative_to(repo_root.resolve()):
            raise ValueError("pilot_source_path_violation")
        digest = sha256(source_path.read_bytes()).hexdigest()
        if digest != record["acquisition"]["sha256"] or source_path.stat().st_size != record["acquisition"]["file_size"]:
            raise ValueError("pilot_source_integrity_failure")
        _validate_acquisition(record, source_path)
        catalog_source = _provenance(record["catalog_provenance"])
        pdf_source = _provenance(record["pdf_provenance"])
        lifecycle = tuple(
            LifecycleEvent(LifecycleKind(item["kind"]), _value(item["value"], item["normalized"], item["display"], catalog_source))
            for item in record["lifecycle"]
        )
        relations = tuple(
            DocumentRelation(RelationKind(item["relation"]), stable_ids[item["source"]], stable_ids[item["target"]], catalog_source)
            for item in record["relations"]
        )
        effects = tuple(
            ProvisionEffect(
                RelationKind(item["relation"]),
                stable_ids[item["source"]],
                stable_ids[item["target"]],
                item["exact_target"],
                item["exact_source_text"],
                SourceProvenance(
                    pdf_source.kind,
                    pdf_source.reference,
                    pdf_source.verified_at,
                    pdf_source.immutable_source_identity,
                    item["page_number"],
                    item["exact_source_text"],
                ),
            )
            for item in record["provision_effects"]
        )
        publication = record["publication"]
        publication_source = SourceProvenance(
            pdf_source.kind,
            pdf_source.reference,
            pdf_source.verified_at,
            pdf_source.immutable_source_identity,
            publication["page_number"],
            publication["source_value"],
        )
        result.append(
            CatalogDocument(
                identity,
                record["short_title"],
                tuple(record["aliases"]),
                StatusAssertion(_value(record["status"], record["status_normalized"], record["status"], catalog_source), catalog_source.verified_at),
                record["document_role"],
                record["document_role_label"],
                lifecycle,
                relations,
                effects,
                _value(publication["source_value"], publication["normalized"], publication["display"], publication_source),
                record["catalog_provenance"]["reference"],
                source_path,
                digest,
                record["acquisition"]["page_count"],
                record["preferred"],
                frozenset(record["permissions"]),
            )
        )
    return tuple(result)


def _identity(record: dict) -> LegalDocumentIdentity:
    source = _provenance(record["catalog_provenance"])
    return LegalDocumentIdentity(
        _value(record["document_type"], record["document_type_normalized"], record["document_type"], source),
        _value(record["number"], record["number"], record["number"], source),
        _value(record["year"], record["year"], record["year"], source),
        _value(record["official_title"], record["official_title"].casefold(), record["official_title"], source),
        _value(record["issuer"], record["issuer"].casefold(), record["issuer"], source),
    )


def _value(source_value: str, normalized: str, display: str, source: SourceProvenance) -> VerifiedValue:
    return VerifiedValue(source_value, normalized, display, FieldState.VERIFIED, source)


def _provenance(record: dict) -> SourceProvenance:
    return SourceProvenance(
        SourceKind(record["kind"]),
        record["reference"],
        datetime.fromisoformat(record["verified_at"]),
        record.get("immutable_source_identity"),
    )


def _validate_acquisition(record: dict, source_path: Path) -> None:
    acquisition = record.get("acquisition", {})
    cross_check = record.get("cross_check", {})
    required = {
        "path", "retrieval_time", "redirect_chain", "mime_type", "file_size", "sha256",
        "page_count", "source_authority", "reviewer_decision",
    }
    if set(acquisition) < required or not cross_check.get("reference") or "discrepancies" not in cross_check:
        raise ValueError("incomplete_pilot_acquisition")
    datetime.fromisoformat(acquisition["retrieval_time"])
    if not isinstance(acquisition["redirect_chain"], list) or not isinstance(cross_check["discrepancies"], list):
        raise ValueError("invalid_pilot_acquisition")
    with pymupdf.open(source_path) as pdf:
        if pdf.page_count != acquisition["page_count"]:
            raise ValueError("pilot_page_count_mismatch")
        publication = record["publication"]
        publication_text = " ".join(pdf[publication["page_number"] - 1].get_text().split()).casefold()
        if " ".join(publication["source_value"].split()).casefold() not in publication_text:
            raise ValueError("publication_identity_not_found_in_official_pdf")
        for effect in record["provision_effects"]:
            page = pdf[effect["page_number"] - 1]
            selected = " ".join(re.sub(r"\s+", " ", page.get_text()).split()).casefold()
            expected = " ".join(effect["exact_source_text"].split()).casefold()
            if expected not in selected:
                raise ValueError("provision_effect_not_found_in_official_pdf")
