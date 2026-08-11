"""Trusted, code-owned corpus strategy composition root."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryNormalizer:
    normalize_query_reference: Callable[[str], str]
    normalize_metadata_intent: Callable[[str], str]


@dataclass(frozen=True)
class ReferenceParser:
    parse_legal_reference: Callable[..., dict[str, str | None]]
    parse_legal_references: Callable[[str], list[dict[str, object]]]
    parse_bab_reference: Callable[[str], str | None]
    parse_pasal_reference: Callable[..., str | None]
    parse_ayat_reference: Callable[[str], str | None]
    label_keys: Callable[[object], set[str]]


@dataclass(frozen=True)
class NavigationResolver:
    resolve_navigation: Callable[[str], tuple[str, str] | None]


@dataclass(frozen=True)
class CorpusContract:
    schema_version: int
    contract_id: str
    contract_version: int
    contract_fingerprint: str


@dataclass(frozen=True)
class CorpusStrategy:
    corpus_id: str
    normalizer: QueryNormalizer
    references: ReferenceParser
    navigation: NavigationResolver
    proposition_operator: Callable[[str], tuple[str, str, str] | None]
    capability_resolver: Callable[..., object] | None = None
    contract: CorpusContract | None = None
    provenance_adapter: object | None = None
    semantic_validator: Callable[[object, dict[str, object]], tuple[str, ...]] | None = None
    citation_unit_factory: Callable[[object, dict[str, object]], object] | None = None
    source_text_query: Callable[[object, str], object | None] | None = None
    source_text_health: Callable[[object], dict[str, int]] | None = None
    embedding_text_normalizer: Callable[[str], str] | None = None


@dataclass(frozen=True)
class StrategyRegistry:
    """Closed, code-owned strategy map; manifests never select imports."""

    strategies: Mapping[str, CorpusStrategy]

    def require(self, corpus_id: str) -> CorpusStrategy:
        try:
            return self.strategies[corpus_id]
        except KeyError as exc:
            raise ValueError(f"unsupported_corpus_parser:{corpus_id}") from exc


def builtin_strategy_registry() -> StrategyRegistry:
    # This is the only composition boundary allowed to import corpus modules.
    from tjipto.corpora.builtin import BUILTIN_STRATEGIES

    return StrategyRegistry(BUILTIN_STRATEGIES)


def strategy_for(corpus_id: str) -> CorpusStrategy:
    return builtin_strategy_registry().require(corpus_id)
