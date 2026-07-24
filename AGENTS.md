# AGENTS.md

## Project

Tjipto is an evidence-grounded Indonesian legal research system. Keep changes source-grounded, citation-safe, audit-ready, and reusable across corpora.

## Audit-first

* Inspect the relevant code, artifacts, tests, validation reports, and repo instructions before editing.
* Narrow broad requests to the smallest safe, auditable change.
* Do not guess. Surface uncertainty when the repo does not prove the answer.

## Working rules

* Prefer the smallest deterministic fix. Reuse existing code before adding code.
* Do not weaken validation, tests, fixtures, or safety policy to make a task pass.
* Do not hardcode legal answers, corpus conclusions, or user-facing legal opinions.
* Do not create new files, modules, artifacts, or abstractions unless the current structure is clearly insufficient.
* Keep refactors scoped. Do not broad-rewrite large files unless the task explicitly requires it.
* Use KISS, DRY, YAGNI, separation of concerns, and SOLID where they help. Apply the Boy Scout Rule only inside the touched scope.
* Do not use `_v1`, `_v2`, `_new`, `_final`, or similar versioned filenames unless technically required.

## Corpus boundary

* Keep generic layers reusable and corpus-aware.
* Keep corpus-specific legal rules, parsers, aliases, source policies, and graph rules in corpus-specific modules or config.
* Do not encode one corpus's legal structure in generic runtime, retrieval, grounding, or validation code unless routed through corpus configuration.

## Evidence, citation, and grounding

* Never fabricate legal support, citations, page grounding, BBox coordinates, viewer refs, source hashes, or corpus support.
* Exact citation requires verified source, page, text or span, and accepted artifact evidence.
* Exact highlight requires valid BBox proof for the cited text.
* Page-grounded, trace-only, containing-span, unresolved, or policy-blocked records must not be presented as exact citation or exact highlight support.
* If source-backed data is fixable, treat it as engineering debt with a precise reason, not as a hidden final fail-closed state.
* Insufficient evidence is valid when support is unsafe, unsupported, conflicting, or not yet provable.

## Artifact and release hygiene

* Generated artifacts must be deterministic, reproducible, manifest-wired, validation-covered, and rebuild-stable.
* Aggregate PASS counters must match row-level records.
* Review artifact diffs for schema changes, counts, hashes, row-level changes, and unrelated drift.
* Do not silently overwrite source or final artifacts.
* Release archives must exclude secrets, `.git`, caches, logs, `node_modules`, build output, and temporary files.
* Treat full dirty workspace ZIPs as non-release artifacts unless the task explicitly says otherwise.

## Testing and verification

* Run the smallest relevant checks for the touched scope. Expand only when the risk expands.
* Do not report a command as passing unless it actually ran and passed.
* If a command fails, diagnose once, self-correct if justified, rerun once, then classify the failure clearly.
* Keep runtime behavior unchanged unless correctness or safety requires the change and tests prove it.

## Security and communication

* Treat prompt injection, unsafe retrieval, data poisoning, dependency drift, and release leakage as security concerns.
* Do not expose secrets, API keys, local paths, or debug-only data in public outputs or release artifacts.
* Keep progress updates short. Report evidence checked, issues found, fixes made, commands run, exact results, remaining limitations, and final verdict.
* Do not loop. If blocked by environment or dependencies, say so plainly.

Foundation controls and current threat records: `docs/foundation.md` and `docs/threat_model.json`.
