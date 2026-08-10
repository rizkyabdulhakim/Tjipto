from __future__ import annotations

import re


UUD_LEGAL_TOKEN_RE = re.compile(
    r"(?m)^(BAB\s+[IVXLCDM]+[A-Z]?|ATURAN PERALIHAN|ATURAN TAMBAHAN|PEMBUKAAN|Pasal\s+(?:[0-9]+[A-Z]?|[IVX]+)|\([0-9]+\)|UNDANG-?UNDANG DASAR)"
)

UUD_BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\s*([a-z]?)\b", re.IGNORECASE)
UUD_PASAL_RE = re.compile(r"\bpasal[ \t\r\n]*([0-9]+)(?:[ \t\r\n]*([a-z]))?\b", re.IGNORECASE)
UUD_PASAL_WITH_AYAT_RE = re.compile(
    r"\bpasal[ \t\r\n]*([0-9]+)(?:[ \t\r\n]*([a-z]))?\b(?:[ \t\r\n]+ayat\s*\(?\s*([0-9]+)\s*\)?)?",
    re.IGNORECASE,
)
UUD_PASAL_OR_ROMAN_RE = re.compile(r"\bpasal[ \t\r\n]*(?:(\d+)(?:[ \t\r\n]*([a-z]))?|([ivxlcdm]+))\b", re.IGNORECASE)
UUD_PASAL_LETTER_RE = re.compile(r"\bpasal\s+([0-9]+)\s+([a-z])\b", re.IGNORECASE)
UUD_PASAL_SHORTHAND_AYAT_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\s*\(\s*([0-9]+)\s*\)", re.IGNORECASE)
UUD_AYAT_RE = re.compile(r"\bayat\s*\(?\s*([0-9]+)\s*\)?", re.IGNORECASE)
UUD_COMPACT_BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\s+([a-z])\b", re.IGNORECASE)
UUD_METADATA_TOKEN_RE = re.compile(r"[0-9a-z]+", re.IGNORECASE)
UUD_PROPOSITION_OPERATORS = (
    ("memperbolehkan", "permits", "permission"),
    ("mewajibkan", "requires", "obligation"),
    ("melarang", "prohibits", "prohibition"),
    ("mengatur", "regulates", "legal_rule"),
    ("menyebut", "mentions", "textual"),
)


def uud_proposition_operator(text: str) -> tuple[str, str, str] | None:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))
    return next((row for row in UUD_PROPOSITION_OPERATORS if re.search(rf"\b{row[0]}\b", normalized)), None)


def normalize_uud_query_reference(text: str) -> str:
    normalized = UUD_PASAL_LETTER_RE.sub(
        lambda match: f"Pasal {match.group(1)}{match.group(2).upper()}",
        text or "",
    )
    normalized = UUD_PASAL_SHORTHAND_AYAT_RE.sub(
        lambda match: f"Pasal {match.group(1).upper()} ayat ({match.group(2)})",
        normalized,
    )
    normalized = UUD_PASAL_RE.sub(lambda match: f"Pasal {match.group(1)}{(match.group(2) or '').upper()}", normalized)
    normalized = UUD_AYAT_RE.sub(lambda match: f"ayat ({match.group(1)})", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_uud_metadata_intent(text: str) -> str:
    """Normalize Indonesian inflection for configured metadata signals only.

    This deliberately leaves retrieval text and legal references unchanged.  The
    small stemmer is enough to put passive/active spelling variants of a
    metadata verb in the same matching bucket without a synonym table.
    """

    def stem(token: str) -> str:
        token = token.casefold()
        if "andatang" not in token:
            return token
        if token.startswith(("men", "pen")) and len(token) > 5 and token[3:4] in "aiueo":
            token = "t" + token[3:]
        elif token.startswith("di") and len(token) > 4:
            token = token[2:]
        if token.endswith("i") and len(token) > 4:
            token = token[:-1]
        if token.endswith("an") and len(token) > 5:
            token = token[:-2]
        return token

    return " ".join(stem(match.group(0)) for match in UUD_METADATA_TOKEN_RE.finditer(text or ""))


def parse_uud_bab_reference(text: str) -> str | None:
    match = UUD_BAB_RE.search(text or "")
    return f"BAB {match.group(1).upper()}{match.group(2).upper()}".strip() if match else None


def parse_uud_pasal_reference(text: str, *, allow_roman: bool = False) -> str | None:
    pattern = UUD_PASAL_OR_ROMAN_RE if allow_roman else UUD_PASAL_RE
    match = pattern.search(text or "")
    if not match:
        return None
    number = match.group(1) or match.group(3)
    suffix = match.group(2) or ""
    return f"Pasal {number}{suffix.upper()}"


def parse_uud_legal_references(text: str) -> list[dict[str, object]]:
    """Parse all Pasal references and retain their source character ranges."""
    rows: list[dict[str, object]] = []
    for match in UUD_PASAL_WITH_AYAT_RE.finditer(text or ""):
        reference = f"Pasal {match.group(1)}{(match.group(2) or '').upper()}"
        if match.group(3):
            reference = f"{reference} ayat ({match.group(3)})"
        rows.append(
            {
                "reference": reference,
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return rows


def matches_uud_contextual_reference(text: str, reference: object) -> bool:
    """Validate one article/paragraph reference inside its source clause."""
    expected_rows = parse_uud_legal_references(str(reference or ""))
    if len(expected_rows) != 1:
        return False
    expected = str(expected_rows[0]["reference"])
    expected_article, _, expected_ayat = expected.partition(" ayat ")
    actual_rows = parse_uud_legal_references(text)
    if not any(str(row["reference"]).partition(" ayat ")[0] == expected_article for row in actual_rows):
        return False
    if not expected_ayat:
        return True
    return any(f"({number})" == expected_ayat for number in UUD_AYAT_RE.findall(text or ""))


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
    if re.search(
        r"\bpasal\s+(?:apa\s+)?berikutnya\s+(?:setelah|sesudah)\b|"
        r"\b(?:setelah|sesudah)\s+pasal\s+\d+[a-z]?\s+(?:pasal\s+berapa|apa)\b",
        normalized,
    ):
        return target, "next"
    if re.search(
        r"\b(?:pasal\s+)?sebelumnya\s+sebelum\b|"
        r"\b(?:ketentuan|bagian|pasal)\s+sebelum\s+pasal\s+\d+[a-z]?\b",
        normalized,
    ):
        return target, "previous"
    return None


def query_normalizer():
    from tjipto.corpora.strategy import QueryNormalizer

    return QueryNormalizer(
        normalize_query_reference=normalize_uud_query_reference,
        normalize_metadata_intent=normalize_uud_metadata_intent,
    )


def reference_parser():
    from tjipto.corpora.strategy import ReferenceParser

    return ReferenceParser(
        parse_legal_reference=parse_uud_legal_reference,
        parse_legal_references=parse_uud_legal_references,
        parse_bab_reference=parse_uud_bab_reference,
        parse_pasal_reference=parse_uud_pasal_reference,
        parse_ayat_reference=parse_uud_ayat_reference,
        label_keys=uud_label_keys,
    )


def navigation_resolver():
    from tjipto.corpora.strategy import NavigationResolver

    return NavigationResolver(
        resolve_navigation=resolve_uud_navigation,
    )
