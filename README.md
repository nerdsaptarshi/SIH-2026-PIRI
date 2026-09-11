# PIRI — Predictive Infrastructure Risk Intelligence

SIH 2026 PS 103 demo: AI-powered project monitoring and early warning.

## Fresh deployment
1. Push this repository to a new GitHub repo.
2. Create a Render Blueprint from `render.yaml`, or create the web service manually.
3. Ensure the service has `DATABASE_URL` pointing to the Render PostgreSQL database.
4. Build: `pip install -r requirements.txt`
5. Start: `python scripts/seed_and_train.py && uvicorn backend.main:app --host 0.0.0.0 --port $PORT --proxy-headers`
6. Open the Render URL.

The bootstrap script intentionally resets the demo database and creates 120 synthetic projects. This is safe for the demo database only. Production deployment must use authorized PAIMANA/OCMS data and migrations instead.

## Main endpoints
/health
/
/api/dashboard
/api/projects/{code}/prediction
/api/alerts
/api/analytics/summary
/api/analytics/sectors
/api/models/metrics
/api/assistant

## Demo disclosure
Current records are synthetic demonstration data. Production integration is designed for authorized PAIMANA/OCMS access.
