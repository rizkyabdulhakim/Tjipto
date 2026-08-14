"""Bounded rank fusion for recall-oriented retrieval lanes.

The sparse and dense scores live in different domains.  This module keeps
those signals as provenance and fuses only their ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from tjipto.retrieval.bm25 import sparse_index_for_store
from tjipto.retrieval.dense import DenseEmbeddingProvider, dense_search


DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RetrievalHit:
    evidence_id: str
    row: dict
    lane: str
    rank: int
    raw_score: float | None
    score_domain: str
    query_variant: str = "original"
    source_role: str | None = None
    temporal_context: str | None = None
    fused_score: float | None = None
    lane_provenance: tuple[tuple[str, int, float | None, str], ...] = ()


def normalize_hits(rows: Iterable[dict], lane: str, *, query_variant: str = "original") -> tuple[RetrievalHit, ...]:
    hits: list[RetrievalHit] = []
    for rank, row in enumerate(rows, start=1):
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id:
            continue
        provenance = row.get(f"_{lane}_provenance") or {}
        raw_score = provenance.get("raw_score")
        if raw_score is not None:
            try:
                raw_score = float(raw_score)
            except (TypeError, ValueError):
                raw_score = None
        hits.append(
            RetrievalHit(
                evidence_id=evidence_id,
                row=dict(row),
                lane=lane,
                rank=int(provenance.get("rank") or rank),
                raw_score=raw_score,
                score_domain=str(provenance.get("score_domain") or lane),
                query_variant=query_variant,
                source_role=row.get("source_role"),
                temporal_context=row.get("temporal_context"),
            )
        )
    return tuple(hits)


def reciprocal_rank_fusion(
    lanes: Mapping[str, Sequence[RetrievalHit]], *, k: int = DEFAULT_RRF_K, limit: int | None = None
) -> tuple[RetrievalHit, ...]:
    """Fuse ranks without performing arithmetic on heterogeneous raw scores."""
    if k < 1:
        raise ValueError("rrf k must be positive")
    by_id: dict[str, list[RetrievalHit]] = {}
    for lane in sorted(lanes):
        for hit in lanes[lane]:
            by_id.setdefault(hit.evidence_id, []).append(hit)
    fused: list[RetrievalHit] = []
    for evidence_id, hits in by_id.items():
        lane_best: dict[str, RetrievalHit] = {}
        for hit in hits:
            current = lane_best.get(hit.lane)
            if current is None or (hit.rank, hit.evidence_id) < (current.rank, current.evidence_id):
                lane_best[hit.lane] = hit
        ordered = tuple(sorted(lane_best.values(), key=lambda hit: (hit.lane, hit.rank, hit.evidence_id)))
        score = sum(1.0 / (k + hit.rank) for hit in ordered)
        primary = ordered[0]
        provenance = tuple((hit.lane, hit.rank, hit.raw_score, hit.score_domain) for hit in ordered)
        fused.append(
            RetrievalHit(
                evidence_id=evidence_id,
                row=dict(primary.row),
                lane="hybrid",
                rank=0,
                raw_score=None,
                score_domain="rrf",
                query_variant=primary.query_variant,
                source_role=primary.source_role,
                temporal_context=primary.temporal_context,
                fused_score=score,
                lane_provenance=provenance,
            )
        )
    ranked = sorted(fused, key=lambda hit: (-float(hit.fused_score or 0.0), hit.evidence_id))
    return tuple(
        replace(hit, rank=rank)
        for rank, hit in enumerate(ranked[:limit] if limit is not None else ranked, start=1)
    )


def hybrid_search(
    store,
    query: str,
    limit: int = 10,
    *,
    provider: DenseEmbeddingProvider | None = None,
    candidate_limit: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    filters: Mapping[str, object] | None = None,
    preferred_source_role: str | None = None,
) -> dict:
    """Run bounded sparse+dense recall and return publication-safe rows."""
    filters = dict(filters or {})
    # Scope is authoritative.  Fetch the complete verified snapshot when a
    # scope or preferred-role constraint is active so filtering happens before
    # the final cutoff rather than after a lane has discarded valid rows.
    scoped = bool(preferred_source_role or filters)
    snapshot_size = len(getattr(store, "evidence", ()))
    budget = snapshot_size if scoped and snapshot_size else candidate_limit if candidate_limit is not None else limit
    if budget < 1:
        return {"status": "no_results", "route": "hybrid", "matches": (), "reason": "invalid_candidate_budget"}
    sparse_rows = _filter_rows(sparse_index_for_store(store).search(query, budget), filters)
    dense_result = dense_search(store, query, budget, provider=provider, include_provenance=True)
    dense_rows = (
        _filter_rows(tuple(dense_result.get("matches") or ()), filters)
        if dense_result.get("status") == "found"
        else ()
    )
    degraded = dense_result.get("reason") if dense_result.get("status") == "dense_unavailable" else None
    hits = reciprocal_rank_fusion(
        {"bm25": normalize_hits(sparse_rows, "bm25"), "dense": normalize_hits(dense_rows, "dense")},
        k=rrf_k,
        limit=None,
    )
    if preferred_source_role:
        preferred = tuple(hit for hit in hits if hit.source_role == preferred_source_role)
        if preferred:
            hits = preferred
    hits = hits[:limit]
    dense_executed = dense_result.get("status") == "found"
    lane_route = "hybrid" if dense_executed else "bm25"
    result_route = "hybrid" if dense_executed else "hybrid_degraded_sparse"
    matches = []
    for hit in hits:
        row = dict(hit.row)
        lanes = tuple(sorted({lane for lane, *_ in hit.lane_provenance}))
        row["route_sources"] = tuple(dict.fromkeys((lane_route, *lanes))) if lanes else (lane_route,)
        row["candidate_type"] = row.get("candidate_type") or "lexical_candidate"
        row.pop("_bm25_provenance", None)
        row.pop("_dense_provenance", None)
        row.pop("_hybrid_provenance", None)
        matches.append(row)
    return {
        "status": "found" if matches else "no_results",
        "route": result_route,
        "matches": tuple(matches),
        "reason": None if matches else (degraded or "no_results"),
        "retrieval_degraded_reason": degraded,
        "candidate_count": len(hits),
        "hybrid_active": dense_executed,
    }


def _filter_rows(rows: Iterable[dict], filters: Mapping[str, object]) -> tuple[dict, ...]:
    return tuple(
        row
        for row in rows
        if ("status" not in filters or row.get("status") == filters["status"])
        and ("source_role" not in filters or row.get("source_role") == filters["source_role"])
        and ("temporal_context" not in filters or row.get("temporal_context") == filters["temporal_context"])
    )


rrf_fuse = reciprocal_rank_fusion
fuse_ranked_hits = reciprocal_rank_fusion


__all__ = [
    "DEFAULT_RRF_K",
    "RetrievalHit",
    "hybrid_search",
    "normalize_hits",
    "reciprocal_rank_fusion",
    "rrf_fuse",
    "fuse_ranked_hits",
]
