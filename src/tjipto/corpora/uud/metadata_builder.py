from __future__ import annotations


def repair_metadata_graph_edges(edges: list[dict], metadata_assertions: list[dict]) -> list[dict]:
    metadata_id_by_key = {
        (row["evidence_link"]["final_evidence_id"], row["predicate"]): row["metadata_id"]
        for row in metadata_assertions
    }
    for edge in edges:
        evidence_id = edge.get("evidence_link", {}).get("final_evidence_id")
        predicate = str(edge.get("target_id") or "").rsplit("::", 1)[-1]
        metadata_id = metadata_id_by_key.get((evidence_id, predicate))
        if metadata_id:
            edge["target_id"] = metadata_id
    return edges
