from __future__ import annotations

from collections.abc import Iterable

from tjipto.catalog import CatalogDocument
from tjipto.contracts.legal_information import FieldState, LifecycleKind, RelationKind, VerifiedValue


_RELATION_LABELS = {
    RelationKind.AMENDS: "Mengubah",
    RelationKind.AMENDED_BY: "Diubah oleh",
    RelationKind.CONSOLIDATES: "Menggabungkan",
    RelationKind.CONSOLIDATED_BY: "Digabungkan dalam",
    RelationKind.DERIVED_FROM: "Berasal dari",
    RelationKind.DERIVES: "Menjadi dasar bagi",
    RelationKind.REVOKES: "Mencabut",
    RelationKind.REVOKED_BY: "Dicabut oleh",
}

_EFFECT_LABELS = {
    "MODIFIES": "Ketentuan yang diubah",
    "RENAMES": "Penomoran menjadi",
    "RENUMBERED_TO": "Penomoran menjadi",
    "DELETES": "Ketentuan yang dihapus",
    "ADDS": "Ketentuan yang ditambahkan",
    "AMBIGUOUS_OPERATION": "Ketentuan yang diubah dan/atau ditambahkan",
}


def project_legal_document(
    document: CatalogDocument,
    documents: Iterable[CatalogDocument],
    *,
    viewer_target: dict | None = None,
) -> dict:
    """Build the one allowlisted legal-document representation used publicly."""
    by_id = {item.stable_id: item for item in documents}
    lifecycle = {event.kind: _public_value(event.value) for event in document.lifecycle}
    projection = {
        "title": _display_value(document.identity.official_title),
        "legal_identity": _identity_display(document),
        "legal_status": _status_display(document.legal_status.status),
        "legal_status_scope": _status_scope_display(document.legal_status.scope),
        "document_role": document.document_role_label,
        "issuer": _display_value(document.identity.issuer),
        "signatories": _public_value(document.signatories) if document.signatories else None,
        "establishment_date": lifecycle.get(LifecycleKind.ESTABLISHMENT),
        "establishment_place": _public_value(document.establishment_place) if document.establishment_place else None,
        "promulgation_date": lifecycle.get(LifecycleKind.PROMULGATION),
        "effective_date": lifecycle.get(LifecycleKind.EFFECTIVENESS),
        "publication": _public_value(document.publication),
        "official_url": document.official_url,
        "relations": tuple(_public_relation(document, relation, by_id) for relation in document.relations),
        "provision_effects": tuple(_public_effect(effect, by_id) for effect in document.provision_effects),
        "source_annotations": tuple(
            {
                "label": f"Catatan sumber {annotation.marker}",
                "text": f"{annotation.marker} berarti {annotation.meaning}.",
                "source_reference": document.official_url,
                "page_number": annotation.page_number,
            }
            for annotation in document.source_annotations
        ),
    }
    conflict = _public_conflict(document.identity.official_title)
    if conflict is not None:
        projection["official_title_conflict"] = conflict
    if viewer_target is not None:
        projection["viewer_target"] = viewer_target
    return projection


def _identity_display(document: CatalogDocument) -> str | None:
    identity = document.identity
    title = _display_value(identity.official_title)
    number = _display_value(identity.number)
    year = _display_value(identity.year)
    document_type = _display_value(identity.document_type)
    if number and document_type:
        return " ".join(
            part for part in (document_type, f"Nomor {number}", f"Tahun {year}" if year else None) if part
        )
    return title


def _status_display(value: VerifiedValue) -> str:
    if value.state is FieldState.VERIFIED and value.display_value:
        return value.display_value
    if value.state is FieldState.CONFLICTING_SOURCES:
        return "Konflik Sumber"
    return "Belum Diverifikasi"


def _status_scope_display(scope: str) -> str:
    return {
        "document_record": "Record peraturan",
        "parent_record": "Record induk sumber",
    }.get(scope, "Cakupan sumber tidak tersedia")


def _public_value(value: VerifiedValue) -> str | None:
    return value.display_value if value.state in {FieldState.VERIFIED, FieldState.CONFLICTING_SOURCES} else None


def _display_value(value: VerifiedValue) -> str | None:
    return value.display_value if value.display_value else None


def _public_relation(document: CatalogDocument, relation, by_id: dict[str, CatalogDocument]) -> dict:
    source = by_id.get(relation.source_document_id)
    target = by_id.get(relation.target_document_id)
    return {
        "label": _RELATION_LABELS[relation.relation],
        "relation_type": _RELATION_LABELS[relation.relation],
        "source": _identity_display(source) if source else "Dokumen sumber belum tersedia",
        "target": _identity_display(target) if target else "Dokumen target belum tersedia",
        "direction": "Naskah sumber ke naskah terkait",
        "verification_state": "Terverifikasi",
        "source_reference": relation.provenance.reference,
    }


def _public_effect(effect, by_id: dict[str, CatalogDocument]) -> dict:
    target = by_id.get(effect.target_document_id)
    return {
        "label": _EFFECT_LABELS.get(effect.operation, "Ketentuan yang diubah"),
        "target": effect.exact_target,
        "document": _identity_display(target) if target else None,
        "verification_state": "Ruang lingkup naskah" if effect.operation == "AMBIGUOUS_OPERATION" else "Terverifikasi",
        "source_reference": effect.provenance.reference,
        "page_number": effect.provenance.page_number,
    }


def _public_conflict(value: VerifiedValue) -> dict | None:
    if value.state is not FieldState.CONFLICTING_SOURCES or value.resolution is None:
        return None
    return {
        "state": "Terselesaikan" if value.resolution.selected_value is not None else "Belum Terselesaikan",
        "kind": "Perbedaan Nilai Sumber Resmi",
        "values": tuple(
            {
                "value": candidate.display_value,
                "source_authority": candidate.provenance.source_authority,
                "source_reference": candidate.provenance.reference,
                "verified_at": candidate.provenance.verified_at.isoformat(),
            }
            for candidate in value.conflicting_values
        ),
        "reviewer_decision": value.resolution.reviewer_decision,
        "legal_basis": value.resolution.legal_basis,
    }
