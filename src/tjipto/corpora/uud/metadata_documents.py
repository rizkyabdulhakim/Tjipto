from __future__ import annotations

from hashlib import sha256
import re

from tjipto.corpora.uud.structure_builder import compact


AMENDMENT_FIELD_STATUSES = {
    "date": "grounded",
    "institution": "grounded",
    "ln_tln": "not_found_in_source",
    "official_title": "not_found_in_source",
    "penetapan": "grounded",
    "pengundangan": "not_found_in_source",
    "place": "grounded",
    "promulgation": "not_found_in_source",
    "signatories": "grounded",
    "source_publication": "not_applicable",
}

CHAIR_TOKEN = "Ketua,"  # nosec B105
VICE_CHAIR_TOKEN = "Wakil"  # nosec B105
CHAIR_ROLE = "Ketua"
VICE_CHAIR_ROLE = "Wakil Ketua"


def build_document_metadata(source_documents: dict[str, dict], metadata_grounding: list[dict]) -> list[dict]:
    block_by_role = {row["source_role"]: row for row in metadata_grounding}
    rows = [
        _amendment_document_metadata(source, block_by_role[source["source_role"]])
        for source in source_documents.values()
        if source["source_role"].startswith("amendment_")
    ]
    rows.append(_current_document_metadata(source_documents["uud::current_consolidated"], block_by_role["current_consolidated"]))
    rows.append(_original_document_metadata(source_documents["uud::original_historical"]))
    return sorted(rows, key=lambda row: row["document_metadata_id"])


def signatory_source_quotes(text: str, signatories: list[dict] | tuple[dict, ...]) -> list[str]:
    """Pair each parsed signatory with its preceding printed role in source order."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    quotes: list[str] = []
    cursor = 0
    for signatory in signatories:
        name = compact(str(signatory.get("name_text") or ""))
        role = _role_key(str(signatory.get("role_text") or ""))
        for role_index in range(cursor, len(lines)):
            if _role_key(lines[role_index]) != role:
                continue
            name_lines: list[str] = []
            for name_index in range(role_index + 1, len(lines)):
                name_lines.append(lines[name_index])
                joined = compact(" ".join(name_lines))
                if joined == name:
                    quotes.append("\n".join((lines[role_index], *name_lines)))
                    cursor = name_index + 1
                    break
                if not name.startswith(joined):
                    break
            else:
                continue
            if cursor > role_index:
                break
        else:
            raise ValueError(f"signatory_source_name_missing:{signatory.get('name_text')}")
    return quotes


def ordinal_label(source_role: str) -> str:
    return {
        "amendment_1_historical": "Pertama",
        "amendment_2_historical": "Kedua",
        "amendment_3_historical": "Ketiga",
        "amendment_4_historical": "Keempat",
    }[source_role]


def extract_metadata_date(text: str) -> str | None:
    match = re.search(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", text)
    return match.group(0) if match else None


def extract_decision_session(text: str, decision_date: str | None) -> str | None:
    value = " ".join(text.split()).split(", dan mulai berlaku", 1)[0].strip()
    if decision_date:
        value = value.replace(f" tanggal {decision_date}", "")
    value = value.removeprefix("Perubahan tersebut diputuskan dalam ").removesuffix(".").strip()
    return value or None


def _amendment_document_metadata(source: dict, block: dict) -> dict:
    date = extract_metadata_date(block["quoted_text"])
    institution, place = _closing_identity(block["quoted_text"])
    grounding_ref = block["metadata_grounding_id"]
    return {
        "corpus_id": "uud",
        "date": date,
        "document_metadata_id": f"uud_document_metadata::{source['source_role']}",
        "document_type": "uud_amendment_document",
        "enactment": None,
        "evidence_refs": [grounding_ref],
        "field_statuses": dict(AMENDMENT_FIELD_STATUSES),
        "grounded_fields": {},
        "grounding_refs": [],
        "institution": institution,
        "ln_tln": None,
        "official_title": None,
        "officials": [],
        "penetapan": {
            "date_text": date,
            "grounding_refs": [grounding_ref],
            "institution": institution,
            "place": place,
        },
        "pengesahan": None,
        "pengundangan": None,
        "place": place,
        "promulgation": None,
        "runtime_loadable": False,
        "signatories": _signatories(block["quoted_text"]),
        "source_document_id": source["source_document_id"],
        "source_publication": None,
        "source_role": source["source_role"],
        "status": "evidence_grounded_partial_metadata",
        "temporal_context": source["temporal_context"],
    }


def _current_document_metadata(source: dict, block: dict) -> dict:
    institution, title = _source_publication_header(block["quoted_text"])
    return {
        "corpus_id": "uud",
        "date": None,
        "document_metadata_id": "uud_document_metadata::current_consolidated",
        "document_type": "uud_consolidated_source_publication",
        "enactment": None,
        "evidence_refs": [block["metadata_grounding_id"]],
        "field_statuses": {
            "date": "not_found_in_source",
            "institution": "grounded",
            "ln_tln": "not_found_in_source",
            "official_title": "grounded",
            "penetapan": "not_applicable",
            "pengundangan": "not_found_in_source",
            "place": "not_found_in_source",
            "promulgation": "not_found_in_source",
            "signatories": "not_applicable",
            "source_publication": "grounded",
            "decision_date": "not_found_in_source",
            "decision_session": "not_found_in_source",
            "effective_rule": "not_found_in_source",
            "effective_date": "not_found_in_source",
            "promulgation_date": "not_found_in_source",
            "revocation_date": "not_found_in_source",
            "source_anomaly_status": "not_found_in_source",
        },
        "grounded_fields": {},
        "grounding_refs": [],
        "institution": institution,
        "ln_tln": None,
        "official_title": title,
        "officials": [],
        "penetapan": None,
        "pengesahan": None,
        "pengundangan": None,
        "place": None,
        "promulgation": None,
        "runtime_loadable": False,
        "signatories": [],
        "source_document_id": source["source_document_id"],
        "source_publication": {
            "grounding_refs": [block["metadata_grounding_id"]],
            "institution": institution,
            "title_text": title,
        },
        "source_role": source["source_role"],
        "status": "evidence_grounded_partial_metadata",
        "temporal_context": source["temporal_context"],
    }


def _original_document_metadata(source: dict) -> dict:
    return {
        "corpus_id": "uud",
        "date": None,
        "document_metadata_id": "uud_document_metadata::original_historical",
        "document_type": "uud_source_document",
        "enactment": None,
        "evidence_refs": [],
        "field_statuses": {
            "metadata": "not_found_in_source",
            "decision_date": "not_found_in_source",
            "decision_session": "not_found_in_source",
            "effective_rule": "not_found_in_source",
            "effective_date": "not_found_in_source",
            "promulgation_date": "not_found_in_source",
            "revocation_date": "not_found_in_source",
            "source_anomaly_status": "not_found_in_source",
            "source_publication": "not_found_in_source",
        },
        "grounded_fields": {},
        "grounding_refs": [],
        "institution": None,
        "ln_tln": None,
        "official_title": None,
        "officials": [],
        "penetapan": None,
        "pengesahan": None,
        "pengundangan": None,
        "place": None,
        "promulgation": None,
        "runtime_loadable": False,
        "signatories": [],
        "source_document_id": source["source_document_id"],
        "source_publication": None,
        "source_role": source["source_role"],
        "status": "not_found_in_source",
        "temporal_context": source["temporal_context"],
    }


def _closing_identity(text: str) -> tuple[str, str]:
    value = " ".join(text.split())
    place = re.search(r"\bDitetapkan di\s+(.+?)\s+Pada tanggal\b", value, flags=re.IGNORECASE)
    institution = re.search(r"\bMAJELIS PERMUSYAWARATAN RAKYAT REPUBLIK INDONESIA\b", value, flags=re.IGNORECASE)
    if not place or not institution:
        raise ValueError("amendment_closing_identity_not_found")
    return institution.group(0).upper(), place.group(1).strip()


def _source_publication_header(text: str) -> tuple[str, str]:
    value = " ".join(text.split())
    title = re.search(
        r"UNDANG\xadUNDANG DASAR NEGARA REPUBLIK INDONESIA TAHUN 1945 DALAM SATU NASKAH$",
        value,
        flags=re.IGNORECASE,
    )
    if not title:
        raise ValueError("consolidated_source_publication_header_not_found")
    return value[: title.start()].rstrip(), title.group(0)


def _signatories(text: str) -> list[dict]:
    parts = text.split()
    rows: list[dict] = []
    role = None
    name_parts: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        spaced_chair = parts[index : index + 5] == ["K", "e", "t", "u", "a,"]
        marker = token == CHAIR_TOKEN or spaced_chair or (
            token == VICE_CHAIR_TOKEN and index + 1 < len(parts) and parts[index + 1] == CHAIR_TOKEN
        )
        if marker:
            if role and name_parts:
                rows.append(_signatory(" ".join(name_parts), role))
            role = VICE_CHAIR_ROLE if token == VICE_CHAIR_TOKEN else CHAIR_ROLE
            name_parts = []
            index += 2 if token == VICE_CHAIR_TOKEN else 5 if spaced_chair else 1
            continue
        if role:
            name_parts.append(token)
        index += 1
    if role and name_parts:
        rows.append(_signatory(" ".join(name_parts), role))
    return rows


def _signatory(name_text: str, role_text: str) -> dict:
    identity_text = re.sub(r"[^a-z0-9]+", "", compact(name_text).casefold())
    return {
        "name_text": name_text,
        "role_text": role_text,
        "entity_identity": f"uud-person-{sha256(identity_text.encode('utf-8')).hexdigest()}",
        "printed_name_alias": name_text,
    }


def _role_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", compact(value))
