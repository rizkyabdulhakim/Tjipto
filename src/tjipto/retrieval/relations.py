from __future__ import annotations

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text
from tjipto.corpora.source_arbitration import initial_source_role, resolve_source_scope, source_roles_for_query
from tjipto.corpora.parser_dispatch import (
    parse_bab_reference,
    parse_pasal_reference,
    parse_legal_references,
)


def relation_lookup(store, query: str, limit: int = 10) -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    if not has_relation_target(query, strategy=strategy, config=config):
        return ()
    relation = _relation(query, strategy=strategy, config=config)
    if relation is None:
        return ()
    intent = intent_config_for(strategy, config)
    route: dict = intent["relation_routes"].get(relation, {})
    descendants_by_parent, evidence_by_unit = _relation_indexes(store)
    if route.get("mode") == "parent":
        source = _unit(store, query, route["source_unit_type"])
        target = _hierarchy_parent(store, source, route["target_unit_type"]) if source else None
        if source is None or target is None:
            return ()
        row = _evidence_for_unit(store, source, evidence_by_unit, descendants_by_parent)
        if row is None:
            return ()
        return (
            row
            | {
                "legal_relation": {
                    "relation_type": relation,
                    "source_legal_unit_id": source["legal_unit_id"],
                    "source_label": source.get("unit_label"),
                    "target_legal_unit_id": target.get("legal_unit_id"),
                    "target_label": target.get("unit_label"),
                },
            },
        )
    parent = _parent_unit(store, query, route)
    if parent is None:
        return ()
    child_type = route.get("child_unit_type")
    children = [
        row
        for row in store.legal_units
        if row.get("unit_type") == child_type and parent["legal_unit_id"] in (row.get("parent_legal_unit_ids") or ())
    ]
    rows = []
    for child in children:
        row = _evidence_for_unit(store, child, evidence_by_unit, descendants_by_parent)
        if row is None:
            continue
        rows.append(
            row
            | {
                "legal_relation": {
                    "relation_type": relation,
                    "source_legal_unit_id": parent["legal_unit_id"],
                    "source_label": parent.get("unit_label"),
                    "target_legal_unit_id": child.get("legal_unit_id"),
                    "target_label": child.get("unit_label"),
                },
            }
        )
    return tuple(rows[:limit])


def amendment_relation_lookup(store, query: str, *, relation_family: str | None = None) -> tuple[dict, tuple[dict, ...]]:
    """Select amendment relations from persisted graph edges, never a sidecar scan."""
    config = getattr(store, "config", None)
    target = amendment_relation_target(store, query, relation_family=relation_family)
    mode = target.get("mode")
    if mode is None or mode == "unsupported":
        return target, ()
    if mode == "document":
        role = target["role"]
        edge_type = "AMENDED_BY" if role == initial_source_role(config) else "AMENDS"
        source_id = f"source_role::{role}"
        return target, tuple(
            row | {"route_sources": ("document_relation_graph",)}
            for row in store.graph_edges
            if row.get("edge_type") == edge_type and row.get("source_id") == source_id
        )
    relation_types = set(target.get("relation_types") or ())
    role = target.get("role")
    rows = []
    for edge in store.graph_edges:
        if edge.get("edge_type") not in relation_types or not edge.get("relation_id"):
            continue
        relation = edge.get("relation_projection") or {}
        if relation is None or relation.get("runtime_loadable") is not True:
            continue
        if role and relation.get("support_source_role", relation.get("source_role")) != role:
            continue
        if not _article_relation_matches_target(
            store,
            relation,
            target.get("target_citation"),
            target.get("target_citations"),
        ):
            continue
        rows.append(edge | {"route_sources": ("article_amendment_relation_graph",)})
    return target, tuple(rows)


def amendment_relation_target(store, query: str, *, relation_family: str | None = None) -> dict:
    config = getattr(store, "config", None)
    if getattr(config, "strategy", None) is None:
        return {"mode": None}
    strategy = getattr(config, "query_strategy", "generic")
    intent = intent_config_for(strategy, config)
    relation_config = intent.get("document_relation", {})
    if not normalize_intent_text(query):
        return {"mode": None}
    relation_families = relation_config.get("relation_families") or {}
    relation_type = relation_family if relation_family in relation_families else _amendment_relation_type(store, query)
    relation_row = relation_families.get(relation_type, {}) if isinstance(relation_families, dict) else {}
    relation_spec = relation_row if isinstance(relation_row, dict) else {}
    relation_types = tuple(relation_spec.get("relation_types") or ())
    source_scope = resolve_source_scope(query, strategy=strategy, config=config)
    references = parse_legal_references(getattr(config, "corpus_id", ""), query, config=config)
    parsed_citations = tuple(str(row.get("reference") or "") for row in references if row.get("reference"))
    # Generic post-amendment wording ("setelah diubah", "pasca perubahan",
    # etc.) selects the current consolidated source; it is not an article
    # relation request unless an amendment/source is named explicitly.
    if source_scope.state == "generic_post_amendment" and len(parsed_citations) == 1:
        return {"mode": None}
    relation_signal = bool(relation_spec) or contains_intent_phrase(query, relation_config.get("change_terms", ()))
    add_signal = _contains_change_form(query, relation_config.get("add_terms", ()))
    if source_scope.explicit and len(references) == 1 and not relation_signal and not add_signal:
        return {"mode": None}
    mentioned_roles = source_roles_for_query(query, strategy=strategy, config=config)
    amendment_role = next((role for role in mentioned_roles if role.startswith("amendment_")), None)
    amendment_signal = amendment_role in set(getattr(config, "source_roles", ()) or ()) or contains_intent_phrase(
        query, relation_config.get("source_terms", ())
    )
    if (
        amendment_role
        and len(mentioned_roles) > 1
        and contains_intent_phrase(query, intent.get("relation_words", ()))
    ):
        return {"mode": "document", "role": amendment_role, "related_roles": mentioned_roles}
    if relation_type == "RENAME_PROVISION":
        amendment_signal = True
    if (
        len(parsed_citations) == 1
        and _has_renumbering_source(store, parsed_citations[0])
        and (
            relation_signal
            or contains_intent_phrase(query, relation_config.get("target_document_terms", ()))
        )
    ):
        return {
            "mode": "article",
            "role": amendment_role,
            "relation_types": ("RENAMES", "RENUMBERED_TO"),
            "target_citation": parsed_citations[0],
            "target_citations": (parsed_citations[0],),
        }
    target_original = contains_intent_phrase(query, relation_config.get("target_document_terms", ()))
    article_detail = contains_intent_phrase(query, relation_config.get("article_detail_terms", ()))
    source_less_delete = relation_type == "DELETE_OR_REMOVE_PROVISION" and (article_detail or amendment_signal)
    if relation_type == "DELETE_OR_REMOVE_PROVISION" and not source_less_delete:
        return {"mode": None}
    origin_role = initial_source_role(config)
    if not references and amendment_role and origin_role in mentioned_roles:
        return {"mode": "document", "role": amendment_role}
    source_less_article_relation = bool(article_detail and relation_signal)
    if not (relation_signal or add_signal) or (
        not amendment_signal and not source_less_delete and not source_less_article_relation
    ):
        return {"mode": None}
    target_citation = _article_relation_target_citation(
        getattr(config, "corpus_id", None), query, prefer_last=relation_type == "RENAME_PROVISION", config=config
    )
    target_citations = (target_citation,) if relation_type == "RENAME_PROVISION" and target_citation else parsed_citations
    if add_signal:
        if article_detail and not target_citation:
            return {"mode": None}
        relation_types = tuple(
            dict.fromkeys(
                (*tuple(value for value in relation_config.get("schema_only_relation_types", ()) if value not in {"RENAMES", "RENUMBERED_TO"}), "AMBIGUOUS_OPERATION")
            )
        )
        return {
            "mode": "article",
            "role": amendment_role,
            "relation_types": relation_types,
            "target_citation": target_citation,
            "target_citations": target_citations,
        }
    if relation_type == "DELETE_OR_REMOVE_PROVISION" and amendment_role and not article_detail:
        return {
            "mode": "article",
            "role": amendment_role,
            "relation_types": ("DELETES",),
            "target_citation": None,
            "target_citations": (),
        }
    if relation_type == "RENAME_PROVISION":
        return {
            "mode": "article",
            "role": amendment_role,
            "relation_types": ("RENAMES", "RENUMBERED_TO"),
            "target_citation": target_citation,
            "target_citations": target_citations,
        }
    article_terms = {
        normalize_intent_text(term)
        for term in tuple(relation_config.get("article_detail_terms", ()) or ())
    }
    legal_object_terms = tuple(
        term
        for term in tuple(intent.get("instrument_legal_object_signals", ()) or ())
        if normalize_intent_text(term) not in article_terms
    )
    detail_terms = tuple(
        dict.fromkeys(
            (
                *tuple(intent.get("instrument_content_signals", ()) or ()),
                *tuple(intent.get("instrument_effect_signals", ()) or ()),
                *tuple(intent.get("instrument_analysis_signals", ()) or ()),
                *legal_object_terms,
            )
        )
    )
    if contains_intent_phrase(query, detail_terms):
        return {"mode": "unsupported"}
    if article_detail:
        if relation_type == "MODIFY_PROVISION":
            relation_types = ("MODIFIES", "ADDS", "AMBIGUOUS_OPERATION")
        return {
            "mode": "article",
            "role": amendment_role,
            "relation_types": relation_types or ("MODIFIES", "ADDS", "DELETES", "AMBIGUOUS_OPERATION"),
            "target_citation": target_citation,
            "target_citations": target_citations,
        }
    if amendment_role and amendment_role.startswith("amendment_"):
        return {"mode": "document", "role": amendment_role}
    if target_original:
        return {"mode": "document", "role": origin_role}
    return {"mode": None}


def _amendment_relation_type(store, query: str) -> str | None:
    config = getattr(store, "config", None)
    relation_config = intent_config_for(getattr(config, "query_strategy", "generic"), config).get("document_relation", {})
    for name, row in (relation_config.get("relation_families") or {}).items():
        explicit_mapping = name == "RENAME_PROVISION" and "menjadi" in (query or "").casefold() and len(
            parse_legal_references(getattr(config, "corpus_id", ""), query, config=config)
        ) >= 2
        if contains_intent_phrase(query, tuple(row.get("terms") or ())) or explicit_mapping:
            return str(name)
    return None


def _contains_change_form(query: str, terms: object) -> bool:
    raw_terms = terms if isinstance(terms, (tuple, list)) else ()
    values = tuple(str(term) for term in raw_terms if isinstance(term, str) and term.strip())
    if contains_intent_phrase(query, values):
        return True
    # Indonesian passive forms such as "ditambahkan" are the configured
    # change term plus a productive suffix, not a new legal concept.
    return contains_intent_phrase(query, tuple(f"{term}kan" for term in values))


def _article_relation_target_citation(corpus_id: str | None, query: str, *, prefer_last: bool = False, config=None) -> str | None:
    if not corpus_id:
        return None
    references = parse_legal_references(corpus_id, query, config=config)
    if not references:
        return None
    return str(references[-1 if prefer_last else 0].get("reference") or "") or None


def _article_relation_matches_target(
    store,
    row: dict,
    target_citation: str | None,
    target_citations: tuple[str, ...] | None = None,
) -> bool:
    requested = tuple(target_citations or ())
    if requested:
        candidates = {
            _normalize_article_target(row.get(key))
            for key in ("target_reference", "new_reference", "target_citation", "old_reference")
            if row.get(key)
        }
        if candidates.intersection({_normalize_article_target(value) for value in requested}):
            return True
    if not target_citation:
        return True
    target = _normalize_article_target(target_citation)
    citation = _normalize_article_target(
        row.get("target_reference")
        or row.get("new_reference")
        or row.get("target_citation")
        or row.get("old_reference")
    )
    if target == citation:
        return True
    unit: dict = next((unit for unit in store.legal_units if unit.get("legal_unit_id") == row.get("target_legal_unit_id")), {})
    return target in {_normalize_article_target(label) for label in [unit.get("unit_label"), *(unit.get("hierarchy") or ())]}


def _normalize_article_target(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("(", "").replace(")", "").split())


def _has_renumbering_source(store, reference: str) -> bool:
    target = _normalize_article_target(reference)
    return any(
        edge.get("edge_type") in {"RENAMES", "RENUMBERED_TO"}
        and _normalize_article_target(
            (edge.get("relation_projection") or {}).get("old_reference")
            or (edge.get("relation_projection") or {}).get("source_label")
        ) == target
        for edge in getattr(store, "graph_edges", ())
    )


def has_relation_target(query: str, *, strategy: str = "generic", config=None) -> bool:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
    if not any(intent[key] for key in ("relation_words", "direct_relation_words", "pasal_parent_words", "relation_child_words")):
        return False
    if _unsupported_relation_requested(query, folded, strategy=strategy, config=config):
        return True
    if _pasal_parent_requested(query, folded, strategy=strategy, config=config):
        return True
    corpus_id = _corpus_id(config)
    if _has_pasal(query, corpus_id) and _route_terms_match(intent, "pasal_children", folded):
        return _child_relation_requested(folded, intent)
    if _has_bab(query, corpus_id) and _route_terms_match(intent, "bab_children", folded):
        return _child_relation_requested(folded, intent)
    return False


def _relation(query: str, *, strategy: str, config=None) -> str | None:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
    if _pasal_parent_requested(query, folded, strategy=strategy, config=config):
        return _route_name(intent, "parent")
    corpus_id = _corpus_id(config)
    if _has_pasal(query, corpus_id) and _route_terms_match(intent, "pasal_children", folded):
        return _route_name(intent, "pasal_children")
    if _has_bab(query, corpus_id) and _route_terms_match(intent, "bab_children", folded):
        return _route_name(intent, "bab_children")
    if _unsupported_relation_requested(query, folded, strategy=strategy, config=config):
        return _route_name(intent, "unsupported")
    return None


def _route_name(intent: dict, requested: str) -> str | None:
    for name, route in intent["relation_routes"].items():
        if requested == route.get("mode"):
            return name
    return None


def _route_terms_match(intent: dict, mode: str, folded: str) -> bool:
    route: dict = next((row for row in intent["relation_routes"].values() if row.get("mode") == mode), {})
    terms = route.get("required_terms") or ()
    return contains_intent_phrase(folded, terms)


def _unsupported_relation_requested(query: str, folded: str, *, strategy: str, config=None) -> bool:
    intent = intent_config_for(strategy, config)
    # A resolved source marker scopes a legal lookup; it is not, by itself, a
    # request to traverse a document relation.
    if resolve_source_scope(query, strategy=strategy, config=config).explicit:
        return False
    corpus_id = _corpus_id(config)
    has_pasal = _has_pasal(query, corpus_id)
    has_bab = _has_bab(query, corpus_id)
    direct_relation = contains_intent_phrase(folded, intent["direct_relation_words"])
    relation_words = contains_intent_phrase(folded, intent["relation_words"])
    unsupported_context = contains_intent_phrase(folded, intent["unsupported_relation_context_words"])
    return (direct_relation and (has_pasal or has_bab or relation_words)) or (
        relation_words and (has_pasal or has_bab or unsupported_context)
    )


def _child_relation_requested(folded: str, intent: dict) -> bool:
    return contains_intent_phrase(folded, intent["relation_child_words"])


def _pasal_parent_requested(query: str, folded: str, *, strategy: str, config=None) -> bool:
    return _has_pasal(query, _corpus_id(config)) and contains_intent_phrase(
        folded, intent_config_for(strategy, config)["pasal_parent_words"]
    )


def _parent_unit(store, query: str, route: dict) -> dict | None:
    if route.get("mode") == "unsupported":
        return None
    parent_type = route.get("parent_unit_type")
    return _unit(store, query, parent_type) if parent_type else None


def _unit(store, query: str, unit_type: str) -> dict | None:
    corpus_id = _corpus_id(getattr(store, "config", None))
    labels = tuple(
        dict.fromkeys(
            label
            for label in (
                parse_pasal_reference(corpus_id, query),
                parse_bab_reference(corpus_id, query),
            )
            if label is not None
        )
    )
    if not labels:
        return None
    scope = resolve_source_scope(query, strategy=getattr(store.config, "query_strategy", "generic"), config=store.config)
    if scope.unresolved:
        return None
    preferred = scope.role
    matches = [
        row
        for row in store.legal_units
        if row.get("unit_label") in labels and row.get("unit_type") == unit_type
    ]
    if scope.explicit:
        return next((row for row in matches if _source_role(row) == preferred), None)
    return next((row for row in matches if _source_role(row) == preferred), None) or (matches[0] if matches else None)


def _hierarchy_parent(store, unit: dict, unit_type: str) -> dict | None:
    parent_ids = set(unit.get("parent_legal_unit_ids") or ())
    if not parent_ids:
        return None
    matches = [row for row in store.legal_units if row.get("legal_unit_id") in parent_ids and row.get("unit_type") == unit_type]
    source_role = unit.get("source_role")
    if source_role is not None:
        return next((row for row in matches if _source_role(row) == source_role), None)
    return matches[0] if matches else None


def _relation_indexes(store) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    descendants_by_parent: dict[str, list[dict]] = {}
    for row in store.legal_units:
        for parent_id in row.get("parent_legal_unit_ids") or ():
            descendants_by_parent.setdefault(parent_id, []).append(row)
    evidence_by_unit: dict[str, list[dict]] = {}
    for row in store.evidence:
        evidence_by_unit.setdefault(row.get("legal_unit_id"), []).append(row)
    return descendants_by_parent, evidence_by_unit


def _source_role(row: dict) -> str | None:
    return row.get("source_role")


def _has_pasal(query: str, corpus_id: str) -> bool:
    return parse_pasal_reference(corpus_id, query) is not None


def _has_bab(query: str, corpus_id: str) -> bool:
    return parse_bab_reference(corpus_id, query) is not None


def _corpus_id(config) -> str:
    return str(getattr(config, "corpus_id", ""))


def _evidence_for_unit(
    store, unit: dict, evidence_by_unit: dict[str, list[dict]], descendants_by_parent: dict[str, list[dict]]
) -> dict | None:
    for candidate in evidence_by_unit.get(unit["legal_unit_id"], ()):
        if candidate.get("status") == "final" and store.bboxes_for(candidate["evidence_id"]):
            return candidate
    for child in descendants_by_parent.get(unit["legal_unit_id"], ()):
        child_candidate = _evidence_for_unit(store, child, evidence_by_unit, descendants_by_parent)
        if child_candidate is not None:
            return child_candidate
    return None
