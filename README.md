# Tjipto Legal Evidence Engine

Local UUD runtime:

```powershell
python -m pip install --require-hashes --upgrade -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m tjipto.runtime.http
```

Optional local LLM chain (9Router primary, Gemini then Groq fallback):

```powershell
# Set the values in the current process (keep API keys out of Git).
$env:TJIPTO_LLM_PROVIDER='openai_compatible'
$env:TJIPTO_LLM_MODEL='ag/gemini-3.7-flash-high'
$env:TJIPTO_LLM_BASE_URL='http://127.0.0.1:20128/v1'
# Set TJIPTO_LLM_API_KEY, TJIPTO_WORDING_API_KEY, and TJIPTO_FALLBACK_LLM_API_KEY locally.
python -m tjipto.runtime.http
```

Keep `.env` local; it is ignored and must never be committed.

Local web app:

```powershell
cd apps/web
npm ci
$env:VITE_TJIPTO_API_BASE='http://localhost:8000'; npm run dev
```

The canonical environment is documented in [docs/foundation.md](docs/foundation.md).

Smoke endpoints:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/uud/ask -Method POST -ContentType 'application/json' -Body '{"query":"Pasal 1 ayat (3)"}'
```
