# Expert QREL v0 annotation protocol

QREL v0 is evaluation data only. Production retrieval and answer paths must never read it. Candidate rows are proposals, not legal ground truth, and do not count toward acceptance.

## Review roles and states

An Indonesian constitutional-law reviewer checks the query, intent, answerability, minimum supporting spans, source role, temporal scope, authority, finality, alternatives, forbidden near-misses, public target policy, and expected abstention or clarification. A different qualified adjudicator resolves disagreements. The allowed flow is `candidate` → `reviewed` → `adjudicated`; only `adjudicated` rows with both review records count as expert QREL. `rejected` rows remain auditable but are not evaluated.

Review records must identify the human reviewer role and use an RFC 3339 UTC `reviewed_at`. Personal names belong in the external review log, not the public dataset. AI generation, automated corpus checks, repository maintainers without the stated legal qualification, and self-review cannot confer `reviewed` or `adjudicated` status.

## Annotation rules

1. Resolve every support ID against the manifest-bound `meaningful_support_units.jsonl` before reading the answer fields.
2. Mark only the smallest sufficient `text_span_ids` as relevant. Put genuinely interchangeable evidence in `alternative_valid_support_ids`; do not treat merely similar text as an alternative.
3. Put neighboring provisions, wrong historical versions, marker/annotation rows, and lexical near-misses in `forbidden_support_ids` when returning them would materially misanswer the query.
4. Use `retrieve` only when the corpus can answer. Use `clarify` when multiple materially different interpretations remain. Use `abstain` for unsupported or out-of-corpus requests.
5. Current and historical text are never interchangeable unless the query expressly permits both. Metadata, structure, instruments, source anomalies, and legal text retain their recorded authority and citation finality.
6. Exact targets must use the selected support geometry. Page-grounded targets remain viewer-only. Typed exclusions have no public target.

## Adjudication and split hygiene

The adjudicator compares the two independent decisions, records the resolved row, and changes `review_status` to `adjudicated`. Acceptance rows are frozen before retrieval tuning. Dev rows may guide diagnosis; acceptance labels must not be imported, embedded, or inspected by production code. Any corpus or manifest identity change requires re-resolution and renewed adjudication before the row counts again.

The evaluator reports only adjudicated rows in metrics. It rejects duplicate case IDs, unresolved or overlapping support sets, invalid minimum spans or targets, stale corpus identities, and a predictions file that contains QREL label fields.
