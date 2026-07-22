from __future__ import annotations

import re

from tjipto.corpora.uud.parser import UUD_LEGAL_TOKEN_RE


INSTRUMENT_UNIT_ROLES = {
    "amendment_recital_record": "instrument_scope",
    "amendment_scope_record": "instrument_scope",
    "instrument_clause_record": "instrument_scope",
    "instrument_closing_record": "instrument_scope",
    "decision_clause_record": "decision_clause",
    "effective_clause_record": "effective_clause",
    "determination_clause_record": "metadata_text",
    "signatory_block_record": "signatory_block",
    "aturan_tambahan_record": "structural_heading",
}

INSTRUMENT_ROLE_CLASSIFICATION = {
    "instrument_scope": "amendment_instrument_text",
    "decision_clause": "decision_clause",
    "effective_clause": "effective_clause",
    "metadata_text": "session_institution_metadata",
    "signatory_block": "signatory_block",
    "source_conflict_trace": "source_conflict_trace",
}

UNIT_SPECIFICITY = {
    "ayat_record": 70,
    "pasal_record": 60,
    "aturan_tambahan_record": 50,
    "aturan_peralihan_record": 50,
    "pembukaan_record": 45,
    "amendment_recital_record": 40,
    "amendment_scope_record": 40,
    "instrument_clause_record": 40,
    "instrument_closing_record": 40,
    "decision_clause_record": 40,
    "effective_clause_record": 40,
    "determination_clause_record": 40,
    "signatory_block_record": 40,
    "bab_record": 10,
}


def role_for_legal_unit(unit: dict) -> str:
    if unit.get("exclusion_ref") == "source_typo_reference::uud_source_typo_reference_00001":
        return "source_conflict_trace"
    unit_type = str(unit.get("unit_type") or "")
    if unit_type in {"pasal_record", "ayat_record", "pembukaan_record"}:
        return "normative_text"
    if unit_type in {"bab_record", "aturan_peralihan_record", "aturan_tambahan_record"}:
        return "structural_heading"
    return INSTRUMENT_UNIT_ROLES.get(unit_type, "needs_review")


def specificity_for_legal_unit(unit: dict) -> int:
    if role_for_legal_unit(unit) == "source_conflict_trace":
        return 80
    return UNIT_SPECIFICITY.get(str(unit.get("unit_type") or ""), 0)


def substantive_structural_unit(unit: dict) -> bool:
    return (
        unit.get("unit_type") == "bab_record"
        and unit.get("source_role") == "current_consolidated"
        and "dihapus" in re.sub(r"\s+", " ", str(unit.get("text") or "")).casefold()
    )


def classification_for_role(role: str) -> str:
    if role == "normative_text":
        return "normative_constitutional_text"
    if role == "structural_heading":
        return "structural_heading"
    if role == "metadata_text":
        return "session_institution_metadata"
    if role == "source_conflict_trace":
        return "source_conflict_trace"
    if role in {"header_footer", "footnote_marker", "separator", "nonlegal_artifact"}:
        return role
    return INSTRUMENT_ROLE_CLASSIFICATION.get(role, "needs_review")


def unreferenced_role(span: dict) -> str:
    text = _clean(span.get("text"))
    if not text:
        return "separator"
    if _is_separator(text):
        return "separator"
    if _is_footnote_marker(text):
        return "footnote_marker"
    if _is_header_footer(span, text):
        return "header_footer"
    if "mulai berlaku" in text.casefold():
        return "effective_clause"
    if UUD_LEGAL_TOKEN_RE.fullmatch(text):
        return "structural_heading"
    return "needs_review"


def _is_separator(text: str) -> bool:
    return bool(text) and not re.search(r"[\w()]", text, re.UNICODE)


def _is_footnote_marker(text: str) -> bool:
    return text in {"*)", "**)", "***)", "****)", ":"}


def _is_header_footer(span: dict, text: str) -> bool:
    upper = text.upper()
    if span.get("source_role") == "current_consolidated" and (span.get("y0") or 0) > 700:
        return True
    return upper in {
        "MAJELIS PERMUSYAWARATAN RAKYAT",
        "SEKRETARIAT JENDERAL",
        "UNDANG-UNDANG DASAR",
        "UNDANG\xadUNDANG DASAR",
        "UNDANG-UNDANG DASAR NEGARA REPUBLIK INDONESIA",
        "UNDANGUNDANG DASAR NEGARA REPUBLIK INDONESIA",
        "UNDANG\xadUNDANG DASAR NEGARA REPUBLIK INDONESIA",
        "NEGARA REPUBLIK INDONESIA TAHUN 1945",
        "TAHUN 1945",
        "DALAM SATU NASKAH",
    }


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()
