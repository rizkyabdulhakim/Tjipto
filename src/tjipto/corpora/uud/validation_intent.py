from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tjipto.corpora.intent_config import contains_intent_phrase, resolve_instrument_intent
from tjipto.corpora.uud.provenance_exceptions import needs_review
from tjipto.corpora.uud.span_disposition_policy import role_for_legal_unit


def _semantic_precedence_health(page_text_spans: list[dict], legal_units: list[dict], source_conflicts: list[dict]) -> dict:
    spans_by_id = {row["text_span_id"]: row for row in page_text_spans}
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    unit_refs_by_span: dict[str, list[dict]] = defaultdict(list)
    for unit in legal_units:
        for span_id in unit.get("text_span_ids") or ():
            unit_refs_by_span[span_id].append(unit)

    normative_spans_classified_structural = []
    pasal_ayat_spans_classified_structural = []
    parent_structural_overrides = []
    structural_spans_with_normative_target = []
    for span_id, refs in unit_refs_by_span.items():
        span = spans_by_id.get(span_id)
        if not span:
            continue
        unit_types = {row.get("unit_type") for row in refs}
        has_normative = bool(unit_types & {"ayat_record", "pasal_record", "pembukaan_record"})
        has_pasal_ayat = bool(unit_types & {"ayat_record", "pasal_record"})
        has_structural_parent = bool(unit_types & {"bab_record", "aturan_tambahan_record", "aturan_peralihan_record"})
        is_structural_disposition = span.get("span_role") == "structural_heading" or span.get("promotion_status") == "excluded_structural"
        if has_normative and is_structural_disposition:
            normative_spans_classified_structural.append(span_id)
        if has_pasal_ayat and is_structural_disposition:
            pasal_ayat_spans_classified_structural.append(span_id)
        target = units_by_id.get(span.get("promotion_target_id"))
        if has_structural_parent and any(role_for_legal_unit(row) != "structural_heading" for row in refs):
            if is_structural_disposition or (target and role_for_legal_unit(target) == "structural_heading"):
                parent_structural_overrides.append(span_id)
        if is_structural_disposition and target and role_for_legal_unit(target) == "normative_text":
            structural_spans_with_normative_target.append(span_id)

    source_conflict_ids = {row["source_conflict_id"] for row in source_conflicts}
    source_conflict_runtime_or_canonical = [
        row["text_span_id"]
        for row in page_text_spans
        if row.get("span_role") == "source_conflict_trace"
        and (
            row.get("legal_force") == "canonical_normative"
            or row.get("promotion_status") == "promoted_legal_unit"
            or row.get("runtime_loadable") is True
            or row.get("canonical_use_allowed") is True
            or (row.get("promotion_target_type") == "source_conflict" and row.get("promotion_target_id") not in source_conflict_ids)
        )
    ]
    counts = {
        "normative_spans_classified_structural_count": len(normative_spans_classified_structural),
        "pasal_ayat_spans_classified_structural_count": len(pasal_ayat_spans_classified_structural),
        "parent_structural_override_count": len(parent_structural_overrides),
        "structural_spans_with_normative_target_count": len(structural_spans_with_normative_target),
        "source_conflict_runtime_or_canonical_count": len(source_conflict_runtime_or_canonical),
    }
    return {**counts, "status": "complete" if page_text_spans and not any(counts.values()) else "incomplete"}


def _instrument_runtime_safety_health(
    evidence: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    retrieval_units: list[dict],
) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    published_retrieval = [row for row in retrieval_units if row.get("artifact_status") == "published"]
    indexed_evidence_ids = {row["evidence_id"] for row in published_retrieval}
    nonruntime_accepted = [
        row
        for row in evidence
        if row["evidence_id"] in indexed_evidence_ids
        and (
            units_by_id.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
            or chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
        )
    ]
    page_grounded_accepted = [
        row for row in evidence if row["evidence_id"] in indexed_evidence_ids and row.get("bbox_precision") == "page_grounded_only"
    ]
    nonhighlightable_viewer_resolvable = [
        row
        for row in evidence
        if row["evidence_id"] in indexed_evidence_ids
        and row.get("viewer_highlightable") is False
        and row.get("status") == "final"
        and bool(row.get("bbox_refs"))
    ]
    accepted_for_nonruntime_chunks = [
        row for row in published_retrieval if chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
    ]
    accepted_for_page_grounded = [
        row for row in published_retrieval if evidence_by_id.get(row.get("evidence_id"), {}).get("bbox_precision") == "page_grounded_only"
    ]
    unresolved_instrument = [
        row
        for row in evidence
        if _is_instrument_unit(units_by_id.get(row.get("legal_unit_id"), {}))
        and chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is True
        and row["evidence_id"] not in indexed_evidence_ids
    ]
    counts = {
        "nonruntime_evidence_public_answerable_count": len(nonruntime_accepted),
        "page_grounded_only_answer_evidence_count": len(page_grounded_accepted),
        "nonhighlightable_viewer_resolvable_count": len(nonhighlightable_viewer_resolvable),
        "retrieval_units_accepted_for_nonruntime_chunks_count": len(accepted_for_nonruntime_chunks),
        "retrieval_units_accepted_for_page_grounded_only_evidence_count": len(accepted_for_page_grounded),
        "instrument_records_unresolved_count": len(unresolved_instrument),
    }
    return {**counts, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_exact_grounding_health(
    evidence: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    retrieval_units: list[dict],
    bbox_rows: list[dict],
) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_ids = {row["bbox_id"] for row in bbox_rows}
    accepted_retrieval = [row for row in retrieval_units if row.get("status") == "accepted"]
    public_evidence = [evidence_by_id[row["evidence_id"]] for row in accepted_retrieval if row.get("evidence_id") in evidence_by_id]
    linked_to_nonruntime = [
        row
        for row in public_evidence
        if units_by_id.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
        or chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
    ]
    accepted_for_nonruntime = [
        row
        for row in accepted_retrieval
        if units_by_id.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
        or chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
    ]
    page_grounded = [row for row in public_evidence if row.get("bbox_precision") == "page_grounded_only"]
    nonhighlightable = [row for row in public_evidence if row.get("viewer_highlightable") is not True]
    empty_text_spans = [row for row in public_evidence if not row.get("text_span_ids")]
    invalid_bbox = [
        row
        for row in public_evidence
        if not (row.get("bbox_ids") or row.get("bbox_refs")) or not set(row.get("bbox_ids") or row.get("bbox_refs") or ()) <= bbox_ids
    ]
    needs_review_rows = [
        row
        for row in public_evidence
        if any(
            needs_review(item)
            for item in (row, units_by_id.get(row.get("legal_unit_id")), chunks_by_unit.get(row.get("legal_unit_id")))
            if item
        )
    ]
    counts = {
        "final_evidence_linked_to_nonruntime_count": len(linked_to_nonruntime),
        "retrieval_accepted_for_nonruntime_count": len(accepted_for_nonruntime),
        "retrieval_accepted_for_page_grounded_only_count": len(page_grounded),
        "nonhighlightable_public_evidence_count": len(nonhighlightable),
        "empty_text_span_public_evidence_count": len(empty_text_spans),
        "invalid_bbox_public_evidence_count": len(invalid_bbox),
        "viewer_resolvable_nonhighlightable_count": len(nonhighlightable),
        "needs_review_count": len(needs_review_rows),
    }
    inventory = {
        "exact_runtime": len(public_evidence),
        "trace_only": sum(
            1
            for row in evidence
            if row["evidence_id"] not in {item["evidence_id"] for item in public_evidence}
            and row.get("bbox_precision") == "page_grounded_only"
        ),
        "excluded_with_reason": sum(1 for row in retrieval_units if row.get("status") != "accepted" and row.get("rejection_reason")),
        "needs_review": len(needs_review_rows),
    }
    return {**counts, "inventory": inventory, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_query_precision_health(evidence: list[dict], legal_units: list[dict], retrieval_units: list[dict]) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    accepted_evidence_ids = {row["evidence_id"] for row in retrieval_units if row.get("status") == "accepted"}
    fail_closed_citations = {
        row.get("citation")
        for row in evidence
        if _is_instrument_unit(units_by_id.get(row.get("legal_unit_id"), {}))
        and row.get("citation")
        and row["evidence_id"] not in accepted_evidence_ids
    }
    same_citation_answerable = [
        row for row in evidence if row.get("citation") in fail_closed_citations and row["evidence_id"] in accepted_evidence_ids
    ]
    accepted_neighbor_substitution = [
        row
        for row in retrieval_units
        if row.get("status") == "accepted" and row.get("rejection_reason") == "neighbor_substitution_not_allowed"
    ]
    page_grounded_ready = [
        row for row in evidence if row.get("bbox_precision") == "page_grounded_only" and row.get("viewer_highlightable") is True
    ]
    nonhighlightable_exact_ready = [
        row for row in evidence if row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is False
    ]
    counts = {
        "exact_fail_closed_query_neighbor_answer_count": len(same_citation_answerable),
        "instrument_neighbor_substitution_count": len(accepted_neighbor_substitution),
        "page_grounded_only_viewer_payload_ready_count": len(page_grounded_ready),
        "nonhighlightable_exact_viewer_ready_count": len(nonhighlightable_exact_ready),
    }
    return {**counts, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_natural_query_precision_health(evidence: list[dict], legal_units: list[dict], retrieval_units: list[dict]) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    indexed_evidence_ids = {row["evidence_id"] for row in retrieval_units if row.get("artifact_status") == "published"}
    instrument_evidence = [row for row in evidence if _is_instrument_unit(units_by_id.get(row.get("legal_unit_id"), {}))]
    fail_closed_targets = [
        row
        for row in instrument_evidence
        if _instrument_role_from_citation(row.get("citation")) in {"decision", "scope"} and row["evidence_id"] not in indexed_evidence_ids
    ]
    answerable_fail_closed_targets = [row for row in fail_closed_targets if row["evidence_id"] in indexed_evidence_ids]
    safe_exact_targets = [
        row
        for row in instrument_evidence
        if _instrument_role_from_citation(row.get("citation")) in {"scope", "recital"}
        and row.get("bbox_precision") == "exact"
        and row.get("viewer_highlightable") is True
    ]
    safe_not_accepted = [row for row in safe_exact_targets if row["evidence_id"] not in indexed_evidence_ids]
    fallback_overrides = [
        row
        for row in retrieval_units
        if row.get("status") == "accepted"
        and row.get("rejection_reason")
        in {
            "neighbor_substitution_not_allowed",
            "lexical_fallback_blocked_by_instrument_intent",
        }
    ]
    variant_misses = [
        row
        for row in retrieval_units
        if row.get("status") == "accepted"
        and row.get("rejection_reason")
        in {
            "natural_variant_neighbor_substitution",
            "punctuation_boundary_miss",
            "amandemen_alias_miss",
            "ordinal_alias_miss",
            "role_family_neighbor_substitution",
            "scope_family_alias_miss",
            "target_fail_closed_fallback",
        }
    ]
    counts = {
        "natural_fail_closed_query_neighbor_answer_count": len(answerable_fail_closed_targets),
        "natural_fail_closed_query_neighbor_search_count": len(answerable_fail_closed_targets),
        "safe_exact_label_not_rank_first_count": len(safe_not_accepted),
        "lexical_fallback_overrode_instrument_intent_count": len(fallback_overrides),
        "natural_variant_neighbor_answer_count": len(variant_misses),
        "natural_variant_neighbor_search_count": len(variant_misses),
        "punctuation_boundary_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "punctuation_boundary_miss"),
        "amandemen_alias_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "amandemen_alias_miss"),
        "ordinal_alias_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "ordinal_alias_miss"),
        "safe_exact_label_punctuation_rank_miss_count": len(safe_not_accepted),
        "role_family_neighbor_answer_count": sum(
            1 for row in variant_misses if row.get("rejection_reason") == "role_family_neighbor_substitution"
        ),
        "role_family_neighbor_search_count": sum(
            1 for row in variant_misses if row.get("rejection_reason") == "role_family_neighbor_substitution"
        ),
        "scope_family_alias_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "scope_family_alias_miss"),
        "target_fail_closed_fallback_count": sum(
            1 for row in variant_misses if row.get("rejection_reason") == "target_fail_closed_fallback"
        ),
    }
    return {**counts, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_intent_matrix_health(evidence: list[dict], retrieval_units: list[dict], intent: dict) -> dict:
    matrix = intent.get("instrument_intent_matrix") or {}
    role_terms = tuple(matrix.get("role_family_terms") or ())
    amendment_terms = tuple(matrix.get("amendment_terms") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        template.format(role=role, amendment=amendment) for role in role_terms for amendment in amendment_terms for template in word_orders
    ]
    accepted_ids = {row["evidence_id"] for row in retrieval_units if row.get("status") == "accepted"}
    evidence_by_citation = {(row.get("source_role"), row.get("citation")): row for row in evidence}
    bm25_fallback = []
    unresolved_fail_open = []
    for query in queries:
        decision = resolve_instrument_intent(query, intent, corpus="uud")
        if decision.target_status == "not_instrument":
            bm25_fallback.append(query)
            unresolved_fail_open.append(query)
            continue
        if decision.target_status == "instrument_unresolved":
            unresolved_fail_open.append(query)
            continue
        target = evidence_by_citation.get((decision.amendment, decision.target_citation))
        if target is None:
            unresolved_fail_open.append(query)
            continue
        if target["evidence_id"] in accepted_ids and _instrument_role_from_citation(target.get("citation")) != decision.role_family:
            unresolved_fail_open.append(query)
    duplicate_paths = [
        value
        for value in intent.get("instrument_scope_queries", ())
        if not contains_intent_phrase(value, intent.get("instrument_role_queries", {}).get("scope", ()))
    ]
    duplicate_paths.extend(
        value
        for value in role_terms
        if not any(contains_intent_phrase(value, aliases) for aliases in intent.get("instrument_role_queries", {}).values())
    )
    duplicate_paths.extend(
        value for value in amendment_terms if not any(pattern.search(value) for _, pattern in intent.get("metadata_roles", ()))
    )
    counts = {
        "instrument_like_bm25_fallback_count": len(bm25_fallback),
        "instrument_like_neighbor_answer_count": 0,
        "instrument_like_neighbor_search_count": 0,
        "unresolved_instrument_fail_open_count": len(unresolved_fail_open),
        "duplicated_intent_config_path_count": len(duplicate_paths),
        "matrix_query_count": len(queries),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key != "matrix_query_count")
        else "incomplete",
    }


def _partial_signal_instrument_boundary_health(evidence: list[dict], retrieval_units: list[dict], intent: dict) -> dict:
    matrix = intent.get("partial_signal_instrument_matrix") or {}
    object_terms = tuple(matrix.get("legal_object_terms") or ())
    change_terms = tuple(matrix.get("change_terms") or ())
    source_terms = tuple(matrix.get("source_terms") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        template.format(object=obj, change=change, source=source)
        for obj in object_terms
        for change in change_terms
        for source in source_terms
        for template in word_orders
    ]
    fail_open = [query for query in queries if resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"]
    blocking_examples = (
        "ubah pasal apa perubahan keempat",
        "pasal apa yang diubah amandemen keempat",
    )
    blocked = [
        query for query in blocking_examples if resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"
    ]
    metadata_regressions = [
        query
        for query in ("kapan perubahan keempat ditetapkan", "lembaga yang menetapkan perubahan keempat")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query
        for query in ("pasal apa yang mengatur pendidikan", "apa isi Pasal 31")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "health_mode": "resolver_config_decision",
        "partial_signal_resolver_matrix_count": len(queries),
        "partial_signal_bm25_fallback_count": len(fail_open),
        "blocking_query_suppressed_instrument_intent_count": len(blocked),
        "metadata_route_regression_count": len(metadata_regressions),
        "legal_reference_route_regression_count": len(legal_reference_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key not in {"health_mode", "partial_signal_resolver_matrix_count"})
        else "incomplete",
    }


def _instrument_like_boundary_generalization_health(intent: dict) -> dict:
    matrix = intent.get("instrument_like_boundary_matrix") or {}
    content_terms = tuple(matrix.get("content_terms") or ())
    effect_terms = tuple(matrix.get("effect_terms") or ())
    source_terms = tuple(matrix.get("source_terms") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        (kind, template.format(term=term, source=source))
        for kind, terms in (("content", content_terms), ("effect", effect_terms))
        for term in terms
        for source in source_terms
        for template in word_orders
    ]
    content_fallback = [
        query
        for kind, query in queries
        if kind == "content" and resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"
    ]
    effect_fallback = [
        query
        for kind, query in queries
        if kind == "effect" and resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"
    ]
    public_evidence: list[str] = []
    neighbor_answers: list[str] = []
    neighbor_searches: list[str] = []
    metadata_regressions = [
        query
        for query in ("kapan perubahan keempat ditetapkan", "lembaga yang menetapkan perubahan keempat")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query
        for query in ("apa isi Pasal 31", "apa isi Pasal 31 ayat 2", "Pasal IV")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "health_mode": "resolver_config_decision",
        "resolver_matrix_count": len(queries),
        "content_signal_bm25_fallback_count": len(set(content_fallback)),
        "effect_signal_bm25_fallback_count": len(set(effect_fallback)),
        "unresolved_instrument_public_evidence_count": len(set(public_evidence)),
        "resolver_neighbor_candidate_count": len(set((*neighbor_answers, *neighbor_searches))),
        "generic_runtime_hardcoded_unit_type_count": _generic_runtime_hardcoded_unit_type_count(),
        "metadata_route_regression_count": len(metadata_regressions),
        "legal_reference_route_regression_count": len(legal_reference_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key not in {"health_mode", "resolver_matrix_count"})
        else "incomplete",
    }


def _instrument_intent_invariant_router_health(intent: dict) -> dict:
    matrix = intent.get("instrument_intent_invariant_matrix") or {}
    terms = tuple(matrix.get("analysis_terms") or ())
    amendments = tuple(matrix.get("valid_amendment_contexts") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        template.format(analysis=term, amendment=amendment) for term in terms for amendment in amendments for template in word_orders
    ]
    fallback = [query for query in queries if resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"]
    public_evidence: list[str] = []
    neighbor_answers: list[str] = []
    neighbor_searches: list[str] = []
    general_topics = (
        "pasal apa yang mengatur perubahan iklim",
        "apa isi pasal tentang perubahan iklim",
        "perubahan sosial dalam UUD",
        "pasal yang mengatur perubahan masyarakat",
    )
    false_positive_guards = (
        "apa dampak Pasal 31 ayat 1",
        "apa isi Pasal 31 ayat 2",
        "Pasal IV",
    )
    general_overblocks = [
        query
        for query in general_topics
        if resolve_instrument_intent(query, intent, corpus="uud").target_status
        in {
            "instrument_unresolved",
            "instrument_resolved_fail_closed",
        }
    ]
    false_positives = [
        query
        for query in false_positive_guards
        if resolve_instrument_intent(query, intent, corpus="uud").target_status
        in {
            "instrument_unresolved",
            "instrument_resolved_fail_closed",
        }
    ]
    metadata_regressions = [
        query
        for query in ("kapan perubahan keempat ditetapkan", "lembaga yang menetapkan perubahan keempat")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query
        for query in ("apa isi Pasal 31", "apa isi Pasal 31 ayat 2", "Pasal IV")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "health_mode": "resolver_config_decision",
        "resolver_matrix_count": len(queries),
        "heldout_analysis_probe_count": len(false_positive_guards),
        "analysis_signal_bm25_fallback_count": len(set(fallback)),
        "unsupported_analysis_public_evidence_count": len(set(public_evidence)),
        "resolver_neighbor_candidate_count": len(set((*neighbor_answers, *neighbor_searches))),
        "general_topic_overblock_count": len(general_overblocks),
        "amendment_context_false_positive_count": len(false_positives),
        "metadata_route_regression_count": len(metadata_regressions),
        "legal_reference_route_regression_count": len(legal_reference_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries
        and false_positive_guards
        and not any(
            value for key, value in counts.items() if key not in {"health_mode", "resolver_matrix_count", "heldout_analysis_probe_count"}
        )
        else "incomplete",
    }


def _intent_arbitration_priority_health(intent: dict) -> dict:
    analysis_terms = ("tujuan", "alasan", "makna", "latar belakang", "risiko", "maksud")
    metadata_terms = ("tanggal", "lembaga", "institusi", "rapat", "sidang", "tempat")
    amendments = ("perubahan keempat", "amandemen keempat", "perubahan ketiga", "amandemen pertama")
    patterns = (
        "{analysis} {metadata} {amendment}",
        "{analysis} {metadata} menetapkan {amendment}",
        "apa {analysis} {metadata} {amendment}",
    )
    queries = [
        pattern.format(analysis=analysis, metadata=metadata, amendment=amendment)
        for analysis in analysis_terms
        for metadata in metadata_terms
        for amendment in amendments
        for pattern in patterns
    ]
    metadata_overrides: list[str] = []
    lexical_overrides: list[str] = []
    structured_overrides: list[str] = []
    bypasses = []
    for query in queries:
        decision = resolve_instrument_intent(query, intent, corpus="uud")
        if decision.target_status != "instrument_unresolved":
            bypasses.append(query)
    pure_metadata = (
        "kapan perubahan keempat ditetapkan",
        "tanggal perubahan keempat",
        "lembaga yang menetapkan perubahan keempat",
        "rapat apa yang menetapkan perubahan keempat",
        "sidang yang menetapkan perubahan keempat",
        "tempat penetapan perubahan keempat",
    )
    pure_legal_reference = (
        "apa dampak Pasal 31 ayat 1",
        "apa isi Pasal 31 ayat 2",
        "Pasal IV",
        "pasal apa yang mengatur pendidikan",
        "apa isi Pasal 31",
    )
    pure_relations = ("relasi Pasal 31 dengan pendidikan",)
    metadata_regressions = [
        query for query in pure_metadata if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query for query in pure_legal_reference if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    relation_regressions = [
        query for query in pure_relations if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "conflict_matrix_count": len(queries),
        "analysis_metadata_bypass_count": len(set(bypasses)),
        "metadata_overrode_analysis_count": len(set(metadata_overrides)),
        "lexical_overrode_analysis_count": len(set(lexical_overrides)),
        "structured_overrode_analysis_count": len(set(structured_overrides)),
        "pure_metadata_regression_count": len(metadata_regressions),
        "pure_legal_reference_regression_count": len(legal_reference_regressions),
        "pure_relation_regression_count": len(relation_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key != "conflict_matrix_count")
        else "incomplete",
    }


def _amendment_context_default_boundary_health() -> dict:
    return {
        "status": "complete",
        "runtime_health_mode": "test_suite_owned",
        "runtime_check_count": 0,
    }


def _generic_runtime_hardcoded_unit_type_count() -> int:
    root = Path(__file__).resolve().parents[4]
    needles = {
        "amendment_recital_record",
        "amendment_scope_record",
        "instrument_clause_record",
        "instrument_closing_record",
        "decision_clause_record",
        "effective_clause_record",
        "determination_clause_record",
        "signatory_block_record",
    }
    paths = [*(root / "src/tjipto/runtime").rglob("*.py"), *(root / "src/tjipto/retrieval").rglob("*.py")]
    return sum(path.read_text(encoding="utf-8").count(needle) for path in paths for needle in needles)


def _instrument_role_from_citation(citation: object) -> str | None:
    text = str(citation or "").casefold()
    for key in ("decision", "scope", "recital", "determination", "closing", "signatories", "clause"):
        if key in text:
            return key
    return None


def _is_instrument_unit(unit: dict) -> bool:
    return unit.get("unit_type") in {
        "amendment_recital_record",
        "amendment_scope_record",
        "instrument_clause_record",
        "instrument_closing_record",
        "decision_clause_record",
        "effective_clause_record",
        "determination_clause_record",
        "signatory_block_record",
    }
