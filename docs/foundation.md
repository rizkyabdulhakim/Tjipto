# R0 foundation

The canonical CI environment is GitHub-hosted `ubuntu-24.04` x86_64, with image metadata retained in CI evidence. It uses CPython 3.12.10, pip 26.1.2, setuptools 83.0.0, Node 24.14.0, npm 11.9.0, and the committed `apps/web/package-lock.json`; its browser is Playwright 1.61.0 Chromium revision 1228. Python extraction remains PyMuPDF 1.27.2.3 / MuPDF 1.27.2; the manifest extractor fingerprint is an immutable artifact input and must not be edited for tooling work.

`requirements.lock` is the canonical base Python lock. `requirements-dense.lock` is its hash-checked dense-runtime extension and is installed only by the dense job after the base lock; together they are the canonical Linux dense environment. The project is then installed editable with `--no-deps --no-build-isolation`. Official Python commands therefore require no `PYTHONPATH` setting. The Node 22/24 conflict was resolved by selecting the clean-install-tested Node 24.14.0 pair; Vite 8 requires Node 22.12 or newer.

CI records commit/tree identity, lock and artifact digests, runner-image metadata, dependency inspection/audit output, gate durations, and release sidecars. The archive is always created from the exact commit, not the worktree.

## Telemetry contract

Telemetry is generic and disabled unless `TJIPTO_TELEMETRY=stderr`. It emits local JSON only; it has no collector, network backend, persistence, or corpus-specific branch. Event names are `corpus_load`, `integrity_failure`, `http_request`, `retrieval_route`, `ci_gate`, and `release_validation`.

Every record is allowlisted by event schema. It may contain only bounded scalar operational attributes; `corpus_id` is an attribute, never a routing branch. Query text, legal text or quotations, tokens, credentials, filesystem paths, and personal data are not accepted attributes. Request IDs are correlation-only and the one intentional high-cardinality attribute. Producers must use fixed route/status/reason/gate values, not user input.

The implemented boundary and deterministic contract tests live in `src/tjipto/telemetry.py` and `tests/test_telemetry_contract.py`. The threat records and future trust boundaries are in [threat_model.json](threat_model.json).

## Meaningful-support ownership

`page_text_spans.jsonl` owns span-level `artifact_status`, `promotion_status`, semantic classification, and source selectors; those fields describe ingestion and promotion, not the final support decision. Authority and citation finality remain owned by `evidence_registry.jsonl`, `metadata_grounding.jsonl`, and `source_conflicts.jsonl`. Exact renderability remains owned by `bbox_registry.jsonl`, `word_bboxes.jsonl`, and raw character lineage. In particular, a rejected span can still have reviewed metadata or source-anomaly support.

`meaningful_support_units.jsonl` is the generated reachability projection across those owners. Its `viewer_eligible`, `highlight_eligible`, and support decision fields are derived references, not new authority or text. The versioned UUD review input at `data/review/uud/meaningful_support_review_decisions.json` owns the two repeated-title adjudications and the reviewed layout-separator exclusion; production code contains no reviewed record identities.

Canonical owner selection is deterministic: the owner must contain the span; match its source document, source role, temporal context, authority kind, and compatible legal force; then the owner with the smallest `text_span_ids` closure wins, with owner ID as the stable equivalent-size tie-breaker. Source-conflict owners derive role and temporal context from their typed anomaly policy. The evaluator implements this contract independently and rejects broader or incompatible substitutions.

Exact support geometry is the union of geometry canonically assigned to the selected spans, with raw-span-contained word alignment used only when it reconstructs the selected quote. Owner-wide geometry is never projected onto a smaller segment. A valid canonical owner, selector, quote, page, and source remain answerable/citable according to authority when overlay geometry is unavailable: the viewer falls back to the page, while highlighting is disabled. Layout exclusions have neither viewer nor highlight capability. No existing artifact field was renamed or removed.

The runtime projection carries only compact, derived viewer rectangles for reviewed meaningful-support units; the audit ledger and raw character lineage remain the authority and geometry owners. `scripts/evaluate_support_reachability.py` independently re-derives every owner, selector, quote, page, and character rectangle before exercising the opaque public viewer target. Retrieval quality is measured by the objective cases in `tests/fixtures/uud/retrieval_cases.jsonl`; evaluation labels remain outside production retrieval.

Answer wording is composed deterministically from `AnswerDecision`, `ClaimSupport`, and the verified evidence projection. The separate cases in `tests/fixtures/uud/answer_cases.jsonl` evaluate factual units and publication contracts without prose snapshots. Optional planner and wording calls remain independently opt-in. Their shared provider owner is configured with `TJIPTO_LLM_PROVIDER`, `TJIPTO_LLM_MODEL`, `TJIPTO_LLM_API_KEY`, optional `TJIPTO_LLM_BASE_URL`, and optional `TJIPTO_LLM_TIMEOUT_SECONDS`; capability-specific `TJIPTO_RESEARCH_PLANNING_*` and `TJIPTO_WORDING_*` values override the shared setting. The model may propose only bounded planning fields or wording over verified claims. It cannot publish facts, citations, viewer targets, authority, temporal context, finality, requirements, or sufficiency; invalid or unavailable output falls back to deterministic behavior.

Bounded research keeps all accepted planner variants but starts the short-lived dense worker at most once per request; remaining variants and retry rounds use the sparse lane. This preserves one true-hybrid retrieval pass without multiplying local model startup and peak-memory cost by the variant budget.
