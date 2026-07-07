# Contributing

Keep changes small, evidence-backed, and covered by the existing checks.

Do not mutate source PDFs, source hashes, evidence quotes, page text, bbox coordinates, or final legal answers unless the task explicitly requires artifact work.

## Official Validation Commands

Backend:

```powershell
python -m compileall -q src tests scripts
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='src;.'; python -m pytest -q -p no:cacheprovider
```

Artifacts:

```powershell
$env:PYTHONPATH='src'; python -m tjipto.corpora.uud_artifact_baseline validate
$env:PYTHONPATH='src'; python -m tjipto.corpora.uud_artifact_baseline rebuild
git diff --exit-code
```

Frontend:

```powershell
cd apps/web
npm ci
npm run typecheck
npm run build
```

Handoff:

```powershell
git archive --format=zip --worktree-attributes HEAD -o tjipto-clean.zip
Expand-Archive tjipto-clean.zip tjipto-clean
python scripts/verify_clean_handoff.py tjipto-clean
```

Do not submit or audit working ZIPs containing `.git`, `node_modules`, `dist`, logs, bytecode, or test caches. Use the clean archive as the audit/release package.
