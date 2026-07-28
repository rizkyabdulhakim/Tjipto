from __future__ import annotations

from tjipto.corpora.strategy import CorpusStrategy, strategy_for


def get_strategy(corpus_id: str, *, config=None) -> CorpusStrategy:
    strategy = getattr(config, "strategy", None)
    return strategy if strategy is not None else strategy_for(corpus_id)


def proposition_operator(corpus_id: str, query: str, *, config=None) -> tuple[str, str, str] | None:
    return get_strategy(corpus_id, config=config).proposition_operator(query)


def normalize_query_reference(corpus_id: str, text: str, *, config=None) -> str:
    return get_strategy(corpus_id, config=config).normalizer.normalize_query_reference(text)


def normalize_metadata_intent(corpus_id: str, text: str, *, config=None) -> str:
    """Return corpus-aware tokens used only for metadata-intent matching."""
    try:
        return get_strategy(corpus_id, config=config).normalizer.normalize_metadata_intent(text)
    except ValueError:
        return " ".join(str(text or "").casefold().split())


def parse_legal_reference(corpus_id: str, text: str, *, allow_roman_pasal: bool = False, config=None) -> dict[str, str | None]:
    return get_strategy(corpus_id, config=config).references.parse_legal_reference(text, allow_roman_pasal=allow_roman_pasal)


def parse_legal_references(corpus_id: str, text: str, *, config=None) -> list[dict[str, object]]:
    return get_strategy(corpus_id, config=config).references.parse_legal_references(text)


def parse_bab_reference(corpus_id: str, text: str, *, config=None) -> str | None:
    return get_strategy(corpus_id, config=config).references.parse_bab_reference(text)


def parse_pasal_reference(corpus_id: str, text: str, *, allow_roman: bool = False, config=None) -> str | None:
    return get_strategy(corpus_id, config=config).references.parse_pasal_reference(text, allow_roman=allow_roman)


def parse_ayat_reference(corpus_id: str, text: str, *, config=None) -> str | None:
    return get_strategy(corpus_id, config=config).references.parse_ayat_reference(text)


def label_keys(corpus_id: str, value: object, *, config=None) -> set[str]:
    return get_strategy(corpus_id, config=config).references.label_keys(value)


def resolve_navigation(corpus_id: str, text: str, *, config=None) -> tuple[str, str] | None:
    return get_strategy(corpus_id, config=config).navigation.resolve_navigation(text)
