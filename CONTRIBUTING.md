# Contributing

Keep changes small, evidence-backed, and covered by the existing checks.

Do not mutate source PDFs, source hashes, evidence quotes, page text, bbox coordinates, or final legal answers unless the task explicitly requires artifact work.

## Official Validation Commands

Install the locked editable environment once from the repository root:

```powershell
python -m pip install --require-hashes --upgrade -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
```

Backend:

```powershell
python -m compileall -q src tests scripts
python -m unittest discover -s tests -v
python -m pytest -q -p no:cacheprovider
```

Artifacts:

```powershell
python -m tjipto.corpora.uud_artifact_baseline validate
python -m tjipto.corpora.uud_artifact_baseline rebuild
git diff --exit-code
```

Static and dependency checks:

```powershell
ruff check src tests scripts
mypy src
bandit -q -r src
python -m pip check
python -m pip_audit --strict -r requirements.lock
```

Frontend:

```powershell
cd apps/web
npm ci
npm run lint
npm run test
npm run typecheck
npm run build
npm audit --omit=dev
```

Handoff:

```powershell
git archive --format=zip --worktree-attributes HEAD -o tjipto-clean.zip
Expand-Archive tjipto-clean.zip tjipto-clean
python scripts/verify_clean_handoff.py tjipto-clean
```

Local `apps/web/node_modules`, `apps/web/dist`, and Python caches are development artifacts; they must never be release artifacts. Use `git archive HEAD`, not a working-tree ZIP, as the audit/release package.
