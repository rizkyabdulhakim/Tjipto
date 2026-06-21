# Tjipto Legal Evidence Engine

Local UUD runtime:

```powershell
$env:PYTHONPATH='src'; python -m tjipto.runtime.http
```

Local web app:

```powershell
cd apps/web
npm install
$env:VITE_TJIPTO_API_BASE='http://localhost:8000'; npm run dev
```

Smoke endpoints:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/uud/ask -Method POST -ContentType 'application/json' -Body '{"query":"Pasal 1 ayat (3)"}'
```
