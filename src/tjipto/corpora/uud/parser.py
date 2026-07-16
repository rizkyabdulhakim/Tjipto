from __future__ import annotations

import re


UUD_LEGAL_TOKEN_RE = re.compile(
    r"(?m)^(BAB\s+[IVXLCDM]+[A-Z]?|ATURAN PERALIHAN|ATURAN TAMBAHAN|PEMBUKAAN|Pasal\s+(?:[0-9]+[A-Z]?|[IVX]+)|\([0-9]+\)|UNDANG-?UNDANG DASAR)"
)

UUD_BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\s*([a-z]?)\b", re.IGNORECASE)
UUD_PASAL_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\b", re.IGNORECASE)
UUD_PASAL_OR_ROMAN_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?|[ivxlcdm]+)\b", re.IGNORECASE)
UUD_PASAL_LETTER_RE = re.compile(r"\bpasal\s+([0-9]+)\s+([a-z])\b", re.IGNORECASE)
UUD_PASAL_SHORTHAND_AYAT_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\s*\(\s*([0-9]+)\s*\)", re.IGNORECASE)
UUD_AYAT_RE = re.compile(r"\bayat\s*\(?\s*([0-9]+)\s*\)?", re.IGNORECASE)
UUD_COMPACT_BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\s+([a-z])\b", re.IGNORECASE)


def normalize_uud_query_reference(text: str) -> str:
    normalized = UUD_PASAL_LETTER_RE.sub(
        lambda match: f"Pasal {match.group(1)}{match.group(2).upper()}",
        text or "",
    )
    normalized = UUD_PASAL_SHORTHAND_AYAT_RE.sub(
        lambda match: f"Pasal {match.group(1).upper()} ayat ({match.group(2)})",
        normalized,
    )
    normalized = UUD_PASAL_RE.sub(lambda match: f"Pasal {match.group(1).upper()}", normalized)
    normalized = UUD_AYAT_RE.sub(lambda match: f"ayat ({match.group(1)})", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def parse_uud_bab_reference(text: str) -> str | None:
    match = UUD_BAB_RE.search(text or "")
    return f"BAB {match.group(1).upper()}{match.group(2).upper()}".strip() if match else None


def parse_uud_pasal_reference(text: str, *, allow_roman: bool = False) -> str | None:
    pattern = UUD_PASAL_OR_ROMAN_RE if allow_roman else UUD_PASAL_RE
    match = pattern.search(text or "")
    return f"Pasal {match.group(1).upper()}" if match else None


def parse_uud_ayat_reference(text: str) -> str | None:
    match = UUD_AYAT_RE.search(text or "")
    return f"({match.group(1)})" if match else None


def parse_uud_legal_reference(text: str, *, allow_roman_pasal: bool = False) -> dict[str, str | None]:
    return {
        "bab": parse_uud_bab_reference(text),
        "pasal": parse_uud_pasal_reference(text, allow_roman=allow_roman_pasal),
        "ayat": parse_uud_ayat_reference(text),
    }


def has_uud_bab_reference(text: str) -> bool:
    return parse_uud_bab_reference(text) is not None


def has_uud_pasal_reference(text: str, *, allow_roman: bool = False) -> bool:
    return parse_uud_pasal_reference(text, allow_roman=allow_roman) is not None


def uud_label_keys(value: object) -> set[str]:
    label = str(value).casefold()
    return {label, UUD_COMPACT_BAB_RE.sub(r"bab \1\2", label)}


def resolve_uud_navigation(text: str) -> tuple[str, str] | None:
    target = parse_uud_pasal_reference(text, allow_roman=True)
    if not target:
        return None
    normalized = normalize_uud_query_reference(text).casefold()
    if re.search(r"\b(?:setelah|sesudah)\b", normalized):
        return target, "next"
    if re.search(r"\bsebelum\b", normalized):
        return target, "previous"
    return None


def parser_adapter():
    from tjipto.corpora.parser_dispatch import CorpusParser

    return CorpusParser(
        normalize_query_reference=normalize_uud_query_reference,
        parse_legal_reference=parse_uud_legal_reference,
        parse_bab_reference=parse_uud_bab_reference,
        parse_pasal_reference=parse_uud_pasal_reference,
        parse_ayat_reference=parse_uud_ayat_reference,
        label_keys=uud_label_keys,
        resolve_navigation=resolve_uud_navigation,
    )
