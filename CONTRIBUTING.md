# Contributing

Keep changes small, evidence-backed, and covered by the existing checks.

Do not mutate source PDFs, source hashes, evidence quotes, page text, bbox coordinates, or final legal answers unless the task explicitly requires artifact work.

Before submitting changes, run:

```powershell
python -m compileall -q src tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
git diff --check
```
