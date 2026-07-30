from __future__ import annotations

from datetime import datetime, timezone

from tjipto.catalog import CatalogDocument
from tjipto.contracts.legal_information import (
    FieldState,
    LegalDocumentIdentity,
    SourceKind,
    SourceProvenance,
    StatusAssertion,
    VerifiedValue,
)


_TITLES = {
    "current_consolidated": ("Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "UUD 1945", True, "current", "Naskah Berlaku"),
    "original_historical": ("Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "UUD 1945 Naskah Asli", False, "historical", "Naskah Historis"),
    "amendment_1_historical": ("Perubahan Pertama Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Pertama UUD 1945", False, "historical", "Naskah Historis"),
    "amendment_2_historical": ("Perubahan Kedua Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Kedua UUD 1945", False, "historical", "Naskah Historis"),
    "amendment_3_historical": ("Perubahan Ketiga Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Ketiga UUD 1945", False, "historical", "Naskah Historis"),
    "amendment_4_historical": ("Perubahan Keempat Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", "Perubahan Keempat UUD 1945", False, "historical", "Naskah Historis"),
}
_YEARS = {
    "current_consolidated": "2002",
    "original_historical": "1945",
    "amendment_1_historical": "1999",
    "amendment_2_historical": "2000",
    "amendment_3_historical": "2001",
    "amendment_4_historical": "2002",
}


def citation_identity(source_role: str) -> tuple[str, str, str]:
    title = _TITLES.get(source_role, _TITLES["current_consolidated"])[0]
    return "Undang-Undang Dasar", "1945", title


def documents(store) -> tuple[CatalogDocument, ...]:
    result = []
    verified_at = datetime(2026, 7, 30, tzinfo=timezone.utc)
    for source in store.source_documents:
        role = str(source.get("source_role"))
        if role not in _TITLES:
            continue
        title, short_title, preferred, document_role, role_label = _TITLES[role]
        provenance = SourceProvenance(
            SourceKind.OFFICIAL_PDF,
            str(source["download_url"]),
            verified_at,
            str(source["sha256"]),
        )
        value = lambda source_value, normalized=None: VerifiedValue(  # noqa: E731
            source_value,
            normalized or source_value.casefold(),
            source_value,
            FieldState.VERIFIED,
            provenance,
        )
        identity = LegalDocumentIdentity(
            value("Undang-Undang Dasar", "undang-undang dasar"),
            value("1945", "1945"),
            value(_YEARS[role], _YEARS[role]),
            value(title, title.casefold()),
            value("Majelis Permusyawaratan Rakyat Republik Indonesia", "mpr ri"),
            value(short_title, role),
        )
        result.append(
            CatalogDocument(
                identity,
                short_title,
                (short_title, title, "konstitusi indonesia"),
                StatusAssertion(value("Berlaku", "applicable"), verified_at),
                document_role,
                role_label,
                (),
                (),
                (),
                VerifiedValue(None, None, None, FieldState.NOT_APPLICABLE),
                str(source["source_page_url"]),
                store.config.source_path(str(source["path"])),
                str(source["sha256"]),
                int(source["page_count"]),
                preferred,
                frozenset({"catalog", "view"}),
            )
        )
    return tuple(result)
