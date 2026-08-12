from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# Shared Indonesian lexical baseline; keep corpus config for legal structure/policy.
STOPWORDS = {
    "adalah",
    "atas",
    "apa",
    "bagaimana",
    "atau",
    "aturan",
    "berapa",
    "boleh",
    "dalam",
    "dan",
    "dari",
    "dengan",
    "di",
    "diatur",
    "dimana",
    "isi",
    "ke",
    "kapan",
    "ketentuan",
    "lembaga",
    "lama",
    "pada",
    "pasal",
    "tentang",
    "menurut",
    "undang",
    "undang-undang",
    "untuk",
    "yang",
    "siapa",
}


def tokens(text: str, *, aliases: dict[str, str] | None = None) -> list[str]:
    expanded: list[str] = []
    for token in TOKEN_RE.findall(text or ""):
        normalized = _normalize_token(token.casefold(), aliases or {})
        expanded.extend(TOKEN_RE.findall(normalized))
    return expanded


def meaningful_tokens(text: str, *, aliases: dict[str, str] | None = None) -> set[str]:
    return {_normalize_token(token, aliases or {}) for token in tokens(text, aliases=aliases) if token not in STOPWORDS and len(token) > 2}


def _normalize_token(token: str, aliases: dict[str, str]) -> str:
    return aliases.get(token, token)


def _document_text(row: dict) -> str:
    return " ".join(
        [
            row.get("quoted_text", ""),
            row.get("citation", ""),
            " ".join(row.get("hierarchy") or []),
        ]
    )


def lexical_search(
    evidence: list[dict],
    query: str,
    limit: int = 10,
    *,
    config=None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    return SparseIndex.build(evidence, config=config, k1=k1, b=b).search(query, limit)


def _freeze(value):
    """Encode nested row data as immutable tagged tuples."""
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(key), _freeze(item)) for key, item in value.items())))
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(_freeze(item) for item in value)))
    return ("scalar", value)


def _thaw(value):
    kind, payload = value
    if kind == "dict":
        return {key: _thaw(item) for key, item in payload}
    if kind == "list":
        return [_thaw(item) for item in payload]
    if kind == "tuple":
        return tuple(_thaw(item) for item in payload)
    if kind == "set":
        return set(_thaw(item) for item in payload)
    return payload


def _canonical_row(row: dict) -> tuple[tuple[str, tuple], ...]:
    """Return the immutable row representation owned by a sparse document."""
    return tuple(sorted((str(key), _freeze(value)) for key, value in row.items()))


def _canonical_row_digest(row: dict) -> str:
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).iterencode(_canonical_row(row)):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SparseDocument:
    """Immutable document state owned by a :class:`SparseIndex`."""

    fields: tuple[tuple[str, tuple], ...]
    terms: tuple[str, ...]
    frequencies: tuple[tuple[str, int], ...]

    @classmethod
    def build(cls, row: dict, terms: tuple[str, ...], frequencies: tuple[tuple[str, int], ...]) -> "SparseDocument":
        return cls(
            fields=_canonical_row(row),
            terms=terms,
            frequencies=frequencies,
        )

    def row(self) -> dict:
        return {key: _thaw(value) for key, value in self.fields}


@dataclass(frozen=True)
class SparseIndex:
    """Immutable BM25 collection statistics for one verified evidence snapshot."""

    identity: str
    aliases: tuple[tuple[str, str], ...]
    k1: float
    b: float
    record_count: int
    documents: tuple[SparseDocument, ...]
    document_frequency: tuple[tuple[str, int], ...]
    avgdl: float

    @classmethod
    def build(
        cls, evidence: list[dict], *, config=None, k1: float = 1.5, b: float = 0.75, identity: str | None = None,
    ) -> "SparseIndex":
        aliases = _lexical_aliases(config)
        alias_items = tuple(sorted(aliases.items()))
        documents: list[SparseDocument] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for row in evidence:
            document_text = _document_text(row)
            doc_terms = tuple(tokens(document_text, aliases=aliases))
            frequencies = Counter(doc_terms)
            documents.append(SparseDocument.build(row, doc_terms, tuple(sorted(frequencies.items()))))
            document_frequency.update(frequencies.keys())
            total_length += len(doc_terms)
        identity = identity or cls.snapshot_identity(evidence, config=config, k1=k1, b=b, aliases=alias_items)
        return cls(
            identity=identity,
            aliases=alias_items,
            k1=k1,
            b=b,
            record_count=len(documents),
            documents=tuple(documents),
            document_frequency=tuple(sorted(document_frequency.items())),
            avgdl=(total_length / len(documents)) if documents else 1.0,
        )

    @staticmethod
    def snapshot_identity(
        evidence: list[dict], *, config=None, k1: float = 1.5, b: float = 0.75,
        aliases: tuple[tuple[str, str], ...] | None = None,
    ) -> str:
        alias_items = aliases if aliases is not None else tuple(sorted(_lexical_aliases(config).items()))
        snapshot = {
            "corpus_id": getattr(config, "corpus_id", None),
            "manifest_digest": getattr(config, "manifest_digest", None),
            "artifact_set_digest": getattr(config, "artifact_set_digest", None),
            "manifest_path": str(getattr(config, "manifest_path", "")),
            "aliases": alias_items,
            "k1": k1,
            "b": b,
            "record_count": len(evidence),
            "records": tuple(sorted((str(row.get("evidence_id", "")), _canonical_row_digest(row)) for row in evidence)),
        }
        return hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        aliases = dict(self.aliases)
        query_terms = tokens(query, aliases=aliases)
        if not query_terms or not self.documents:
            return []
        document_frequency = dict(self.document_frequency)
        total_docs = self.record_count
        scored: list[tuple[float, str, dict]] = []
        for document in self.documents:
            row = document.row()
            doc_terms = document.terms
            frequency_items = document.frequencies
            if not doc_terms:
                continue
            frequencies = dict(frequency_items)
            doc_len = len(doc_terms)
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            if score > 0:
                scored.append((score, row["evidence_id"], _with_coverage(row, query, aliases)))
        results = []
        for rank, (score, _, row) in enumerate(sorted(scored, key=lambda item: (-item[0], item[1]))[:limit], start=1):
            result = dict(row)
            result["_bm25_provenance"] = {
                "retriever": "bm25",
                "raw_score": score,
                "rank": rank,
                "score_domain": "bm25",
            }
            results.append(result)
        return results


def sparse_index_for_store(store, *, k1: float = 1.5, b: float = 0.75) -> SparseIndex:
    """Return the store-scoped index, rebuilding when its verified snapshot changes."""
    index = getattr(store, "_sparse_index", None)
    evidence = store.evidence
    config = store.config
    aliases = tuple(sorted(_lexical_aliases(config).items()))
    cache_key = SparseIndex.snapshot_identity(evidence, config=config, k1=k1, b=b, aliases=aliases)
    if index is None or getattr(store, "_sparse_index_cache_key", None) != cache_key:
        store._sparse_index = SparseIndex.build(evidence, config=config, k1=k1, b=b, identity=cache_key)
        store._sparse_index_cache_key = cache_key
        return store._sparse_index
    return index


def _with_coverage(row: dict, query: str, aliases: dict[str, str]) -> dict:
    query_terms = meaningful_tokens(query, aliases=aliases)
    doc_terms = meaningful_tokens(_document_text(row), aliases=aliases)
    supported = query_terms & doc_terms
    # Coverage is a neutral retrieval signal. Support validation, not BM25,
    # decides whether a candidate can be published.
    required = len(query_terms)
    complete = bool(query_terms) and len(supported) >= required
    return dict(
        row,
        lexical_query_terms=tuple(sorted(query_terms)),
        lexical_supported_terms=tuple(sorted(supported)),
        lexical_term_coverage=(len(supported) / required) if required else 0.0,
        lexical_complete_coverage=complete,
    )


def _lexical_aliases(config) -> dict[str, str]:
    settings: dict = getattr(config, "setting", lambda *_: {})("lexical_normalization", {}) or {}
    return {str(key).casefold(): str(value).casefold() for key, value in dict(settings.get("aliases") or {}).items()}
