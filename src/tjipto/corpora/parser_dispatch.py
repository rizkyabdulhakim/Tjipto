from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tjipto.corpora.uud import parser as uud_parser


DEFAULT_CORPUS_ID = "uud"


@dataclass(frozen=True)
class CorpusParser:
    normalize_query_reference: Callable[[str], str]
    normalize_metadata_intent: Callable[[str], str]
    parse_legal_reference: Callable[..., dict[str, str | None]]
    parse_legal_references: Callable[[str], list[dict[str, object]]]
    parse_bab_reference: Callable[[str], str | None]
    parse_pasal_reference: Callable[..., str | None]
    parse_ayat_reference: Callable[[str], str | None]
    label_keys: Callable[[object], set[str]]
    resolve_navigation: Callable[[str], tuple[str, str] | None]


_PARSERS: dict[str, CorpusParser] = {
    "uud": uud_parser.parser_adapter(),
}


def get_parser(corpus_id: str) -> CorpusParser:
    try:
        return _PARSERS[corpus_id]
    except KeyError as exc:
        raise ValueError(f"unsupported_corpus_parser:{corpus_id}") from exc


def normalize_query_reference(corpus_id: str, text: str) -> str:
    return get_parser(corpus_id).normalize_query_reference(text)


def normalize_metadata_intent(corpus_id: str, text: str) -> str:
    """Return corpus-aware tokens used only for metadata-intent matching."""
    try:
        return get_parser(corpus_id).normalize_metadata_intent(text)
    except ValueError:
        return " ".join(str(text or "").casefold().split())


def parse_legal_reference(corpus_id: str, text: str, *, allow_roman_pasal: bool = False) -> dict[str, str | None]:
    return get_parser(corpus_id).parse_legal_reference(text, allow_roman_pasal=allow_roman_pasal)


def parse_legal_references(corpus_id: str, text: str) -> list[dict[str, object]]:
    return get_parser(corpus_id).parse_legal_references(text)


def parse_bab_reference(corpus_id: str, text: str) -> str | None:
    return get_parser(corpus_id).parse_bab_reference(text)


def parse_pasal_reference(corpus_id: str, text: str, *, allow_roman: bool = False) -> str | None:
    return get_parser(corpus_id).parse_pasal_reference(text, allow_roman=allow_roman)


def parse_ayat_reference(corpus_id: str, text: str) -> str | None:
    return get_parser(corpus_id).parse_ayat_reference(text)


def label_keys(corpus_id: str, value: object) -> set[str]:
    return get_parser(corpus_id).label_keys(value)


def resolve_navigation(corpus_id: str, text: str) -> tuple[str, str] | None:
    return get_parser(corpus_id).resolve_navigation(text)
