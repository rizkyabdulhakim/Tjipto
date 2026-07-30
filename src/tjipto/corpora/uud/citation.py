from __future__ import annotations

from tjipto.contracts.legal_information import CitationUnit
from tjipto.corpora.uud.catalog import citation_identity


def citation_unit(store, row: dict[str, object]) -> CitationUnit | None:
    role = str(row.get("source_role") or "")
    source_document_id = str(row.get("source_document_id") or "")
    try:
        document_type, year, title = citation_identity(role)
    except ValueError:
        return None
    source: dict[str, object] = next(
        (
            item
            for item in store.source_documents
            if item.get("source_role") == role or (source_document_id and item.get("source_document_id") == source_document_id)
        ),
        {},
    )
    official_url = str(source.get("source_page_url") or source.get("download_url") or "")
    if not official_url:
        return None
    provision = str(row.get("canonical_label") or row.get("printed_name") or row.get("label") or "").strip() or None
    raw_pages = row.get("page_numbers")
    pages = tuple(raw_pages) if isinstance(raw_pages, (list, tuple)) else ()
    return CitationUnit(
        evidence_key=str(row.get("evidence_id") or row.get("metadata_grounding_id") or row.get("relation_id") or f"{role}:{provision}"),
        document_type=document_type,
        number=None,
        year=year,
        official_title=title,
        publication=None,
        provision=provision,
        page=int(pages[0]) if pages else None,
        official_url=official_url,
        authority=str(row.get("authority_kind") or "source_text"),
        citation_final=row.get("citation_final") is True,
    )
