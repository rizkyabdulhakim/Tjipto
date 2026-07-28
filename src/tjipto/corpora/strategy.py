"""Trusted, code-owned corpus strategy composition root."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tjipto.corpora.uud import parser as uud_parser


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


@dataclass(frozen=True)
class CorpusStrategy:
    corpus_id: str
    parser: CorpusParser
    proposition_operator: Callable[[str], tuple[str, str, str] | None]


_STRATEGIES = {"uud": CorpusStrategy("uud", uud_parser.parser_adapter(), uud_parser.uud_proposition_operator)}


def strategy_for(corpus_id: str) -> CorpusStrategy:
    try:
        return _STRATEGIES[corpus_id]
    except KeyError as exc:
        raise ValueError(f"unsupported_corpus_parser:{corpus_id}") from exc
