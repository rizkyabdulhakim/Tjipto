# Tjipto Repo Guardrails

- Current runtime scope is UUD only. Do not add a new corpus, broad refactor, or runtime layer without an explicit task.
- Backend code lives under `src/tjipto`; the local web frontend lives under `apps/web`.
- Runtime code must be evidence-grounded: no fake citations, bbox coordinates, hashes, PDF proof, or viewer references.
- Keep data artifacts and source PDFs unchanged unless the task explicitly says to regenerate or audit them.
- Keep runtime and pipeline concerns separate. Runtime reads final artifacts through registry/manifest-backed contracts; pipeline or validation work stays out of serving paths.
- Protect provenance rules: do not promote manual-review or process artifacts into active runtime evidence.
