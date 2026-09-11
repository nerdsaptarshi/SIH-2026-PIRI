# PIRI SIH 2026 — Demo Polish Pack

This pack contains:
- `frontend/index.html`: polished demo dashboard with one-call dashboard loading, project risk table, sector risk, early warnings, explainable AI panel and assistant.
- `backend/main.py`: optimized backend with a new `/api/dashboard` batch endpoint and batch scoring for summary/sectors/alerts.

## Deployment order

1. Replace `frontend/index.html` in the repository with the supplied file.
2. Replace `backend/main.py` with the supplied file.
3. Commit both together.
4. Wait for Render to redeploy.
5. Hard refresh with Ctrl+Shift+R.

## Important
The current prototype uses synthetic demonstration data. Do not claim the live demo is trained on restricted PAIMANA/OCMS records. The production architecture is designed for authorized integration.
