from __future__ import annotations

from tjipto.corpora.strategy import CorpusParser, strategy_for


def get_parser(corpus_id: str) -> CorpusParser:
    return strategy_for(corpus_id).parser


def proposition_operator(corpus_id: str, query: str) -> tuple[str, str, str] | None:
    return strategy_for(corpus_id).proposition_operator(query)


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
