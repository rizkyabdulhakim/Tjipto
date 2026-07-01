from __future__ import annotations

from types import ModuleType

from tjipto.corpora.uud import parser as uud_parser


DEFAULT_CORPUS_ID = "uud"

_PARSERS: dict[str, ModuleType] = {
    "uud": uud_parser,
}


def get_parser(corpus_id: str) -> ModuleType:
    try:
        return _PARSERS[corpus_id]
    except KeyError as exc:
        raise ValueError(f"unsupported_corpus_parser:{corpus_id}") from exc


def normalize_query_reference(corpus_id: str, text: str) -> str:
    return get_parser(corpus_id).normalize_uud_query_reference(text)


def parse_legal_reference(corpus_id: str, text: str, *, allow_roman_pasal: bool = False) -> dict[str, str | None]:
    return get_parser(corpus_id).parse_uud_legal_reference(text, allow_roman_pasal=allow_roman_pasal)


def parse_bab_reference(corpus_id: str, text: str) -> str | None:
    return get_parser(corpus_id).parse_uud_bab_reference(text)


def parse_pasal_reference(corpus_id: str, text: str, *, allow_roman: bool = False) -> str | None:
    return get_parser(corpus_id).parse_uud_pasal_reference(text, allow_roman=allow_roman)


def parse_ayat_reference(corpus_id: str, text: str) -> str | None:
    return get_parser(corpus_id).parse_uud_ayat_reference(text)


def label_keys(corpus_id: str, value: object) -> set[str]:
    return get_parser(corpus_id).uud_label_keys(value)
