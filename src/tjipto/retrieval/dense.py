from __future__ import annotations

from tjipto.evidence.store import EvidenceStore


def dense_search(store: EvidenceStore, query: str, limit: int = 10) -> dict:
    configured = bool(store.config.manifest.get("dense_retrieval"))
    if not configured:
        return {
            "status": "dense_unavailable",
            "route": "dense_unavailable",
            "matches": (),
            "reason": "not_configured",
        }
    return {
        "status": "dense_unavailable",
        "route": "dense_unavailable",
        "matches": (),
        "reason": "not_implemented",
    }
