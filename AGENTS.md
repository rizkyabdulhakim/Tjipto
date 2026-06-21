# AGENTS.md

Behavioral guidelines for AI coding agents working on Tjipto.

These instructions are intentionally short. Merge them with the user request and the actual repository state. They are not a roadmap, product scope document, or substitute for reading the code.

## 1. Think Before Coding

Do not assume. Inspect the relevant files first.

Before editing:
- understand the current behavior
- identify what must not be touched
- choose the smallest safe change
- surface uncertainty instead of guessing

If the request is broad, narrow it to a safe, auditable change.

## 2. Simplicity First

Use YAGNI.

Prefer deletion, consolidation, and existing code over new code.

Do not add speculative features, one-off abstractions, new folders, or dependencies unless the task proves they are needed.

Every changed line should trace back to the task.

## 3. Evidence First

Tjipto is an evidence-grounded Indonesian legal research system, not a generic chatbot.

Do not fabricate legal support, citations, page grounding, bbox coordinates, source hashes, PDF links, viewer refs, or corpus support.

Do not mutate source PDFs or evidence artifacts unless the task explicitly approves artifact work.

Insufficient evidence is a valid result. Preserve it instead of inventing support.

## 4. Surgical Changes

Respect the existing repository layout.

Do not create duplicate source roots, typo folders, temporary folders, or broad refactors.

Runtime code must not import extraction/pipeline code.

Frontend code must not read legal data artifacts directly.

HTTP/UI code must not duplicate retrieval logic.

Public answers and UI payloads must use accepted evidence only.

## 5. Goal-Driven Execution

Define how the change will be verified before declaring success.

Run the smallest relevant checks:
- backend: compile, tests, diff check
- frontend: typecheck, build, audit
- web: HTTP smoke and browser smoke when available
- artifacts: artifact/provenance validators when touched

Do not report checks as passed if they were not run.

End with changed files, validation results, artifact/data safety, known limitations, and final git status.
