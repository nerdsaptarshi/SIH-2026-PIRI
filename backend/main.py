from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import pandas as pd
import json
import os
from datetime import datetime

from .db import get_db, Project, Prediction
from .schemas import ProjectOut, PredictionOut, AlertOut, ChatRequest, ChatResponse
from .model_pipeline import load_models, predict, explain_project, FEATURES
from .config import MODEL_DIR

app = FastAPI(title="PIRI AI Infrastructure Monitoring API", version="1.1.0")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = None
MODEL_VERSION = "untrained"


@app.on_event("startup")
def startup():
    global MODELS, MODEL_VERSION
    try:
        MODELS = load_models()
        MODEL_VERSION = (MODEL_DIR / "version.txt").read_text().strip()
    except Exception:
        MODELS = None


def project_row(p):
    return pd.DataFrame([{k: getattr(p, k) for k in FEATURES}])


def portfolio_scores(projects):
    """Batch-score the entire portfolio in four model calls instead of
    running predict() separately for every project."""
    if not MODELS or not projects:
        return []

    frame = pd.DataFrame([{k: getattr(p, k) for k in FEATURES} for p in projects])

    cost_prob = MODELS["cost"].predict_proba(frame[FEATURES])[:, 1]
    delay_prob = MODELS["delay"].predict_proba(frame[FEATURES])[:, 1]
    cost_pct = MODELS["cost_pct"].predict(frame[FEATURES])

    scores = []
    for i, p in enumerate(projects):
        pc = float(cost_prob[i])
        pdly = float(delay_prob[i])
        cp = float(cost_pct[i])
        score = round(100 * (
            0.45 * pc +
            0.40 * pdly +
            0.15 * min(max(cp / 30, 0), 1)
        ), 1)
        level = "High" if score >= 70 else "Medium" if score >= 45 else "Low"

        scores.append({
            "project_code": p.project_code,
            "cost_overrun_probability": round(pc, 4),
            "delay_probability": round(pdly, 4),
            "cost_overrun_pct": round(cp, 2),
            "risk_score": score,
            "risk_level": level,
        })
    return scores


def project_payload(p, pred):
    return {
        "project_code": p.project_code,
        "name": p.name,
        "sector": p.sector,
        "ministry": p.ministry,
        "original_cost_cr": p.original_cost_cr,
        "revised_cost_cr": p.revised_cost_cr,
        "expenditure_cr": p.expenditure_cr,
        "physical_progress_pct": p.physical_progress_pct,
        "schedule_progress_pct": p.schedule_progress_pct,
        "cost_growth_pct": p.cost_growth_pct,
        "milestones_delayed": p.milestones_delayed,
        "clearance_pending": p.clearance_pending,
        **pred,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODELS is not None,
        "model_version": MODEL_VERSION,
    }


@app.get("/api/projects", response_model=list[ProjectOut])
def projects(db: Session = Depends(get_db), q: str = "", sector: str = "", risk: str = ""):
    ps = db.query(Project).filter(Project.status == "Ongoing").all()
    if q:
        ps = [p for p in ps if q.lower() in (p.name + " " + p.project_code + " " + p.ministry).lower()]
    if sector:
        ps = [p for p in ps if p.sector == sector]
    if risk and MODELS:
        scored = {x["project_code"]: x for x in portfolio_scores(ps)}
        ps = [p for p in ps if scored.get(p.project_code, {}).get("risk_level") == risk]
    return ps


@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)):
    """Single batch endpoint for the demo dashboard.
    It returns KPIs, sector risk, alerts and the fully scored portfolio
    in one request."""
    if not MODELS:
        raise HTTPException(503, "Models not trained.")

    ps = db.query(Project).filter(Project.status == "Ongoing").all()
    scored = portfolio_scores(ps)
    by_code = {x["project_code"]: x for x in scored}

    portfolio = [project_payload(p, by_code[p.project_code]) for p in ps]

    summary = {
        "projects": len(ps),
        "original_cost_cr": round(sum(p.original_cost_cr for p in ps), 2),
        "revised_cost_cr": round(sum(p.revised_cost_cr for p in ps), 2),
        "expenditure_cr": round(sum(p.expenditure_cr for p in ps), 2),
        "high_risk": sum(1 for x in scored if x["risk_level"] == "High"),
        "medium_risk": sum(1 for x in scored if x["risk_level"] == "Medium"),
        "low_risk": sum(1 for x in scored if x["risk_level"] == "Low"),
    }

    grouped = {}
    for p in ps:
        grouped.setdefault(p.sector, []).append(by_code[p.project_code]["risk_score"])

    sectors = [
        {
            "sector": sector,
            "projects": len(values),
            "avg_risk": round(sum(values) / len(values), 1),
            "high_risk": sum(1 for v in values if v >= 70),
        }
        for sector, values in grouped.items()
    ]
    sectors.sort(key=lambda x: x["avg_risk"], reverse=True)

    alerts = []
    for p in ps:
        x = by_code[p.project_code]
        if x["risk_score"] >= 70:
            signal = "Cost escalation + schedule risk" if x["cost_overrun_probability"] >= x["delay_probability"] else "Schedule delay risk"
            action = (
                "Cost-driver review and milestone recovery plan"
                if x["cost_overrun_probability"] >= x["delay_probability"]
                else "Milestone recovery plan and implementation review"
            )
            alerts.append({
                "project_code": p.project_code,
                "project_name": p.name,
                "severity": "CRITICAL" if x["risk_score"] >= 85 else "HIGH",
                "signal": signal,
                "probability": round(max(x["cost_overrun_probability"], x["delay_probability"]), 3),
                "recommended_action": action,
            })
    alerts.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "sectors": sectors,
        "alerts": alerts[:25],
        "projects": portfolio,
    }


@app.get("/api/projects/{code}", response_model=ProjectOut)
def project(code: str, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.project_code == code).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@app.post("/api/projects/{code}/predict", response_model=PredictionOut)
def project_prediction(code: str, db: Session = Depends(get_db)):
    if not MODELS:
        raise HTTPException(503, "Models not trained. Run scripts/seed_and_train.py first.")

    p = db.query(Project).filter(Project.project_code == code).first()
    if not p:
        raise HTTPException(404, "Project not found")

    row = project_row(p)
    pc, pdly, cost_pct, delay_m, score, level = predict(MODELS, row)
    drivers = explain_project(MODELS, row)

    rec = {
        "project_code": code,
        "cost_overrun_probability": round(pc, 4),
        "delay_probability": round(pdly, 4),
        "risk_score": score,
        "risk_level": level,
        "top_drivers": drivers,
        "model_version": MODEL_VERSION,
    }

    db.add(Prediction(
        project_code=code,
        cost_overrun_probability=pc,
        delay_probability=pdly,
        risk_score=score,
        risk_level=level,
        top_drivers=json.dumps(drivers),
        model_version=MODEL_VERSION,
    ))
    db.commit()
    return rec


@app.get("/api/projects/{code}/prediction")
def get_prediction(code: str, db: Session = Depends(get_db)):
    p = db.query(Prediction).filter(Prediction.project_code == code).order_by(Prediction.id.desc()).first()
    if not p:
        return project_prediction(code, db)
    return {
        "project_code": p.project_code,
        "cost_overrun_probability": p.cost_overrun_probability,
        "delay_probability": p.delay_probability,
        "risk_score": p.risk_score,
        "risk_level": p.risk_level,
        "top_drivers": json.loads(p.top_drivers),
        "model_version": p.model_version,
    }


@app.get("/api/alerts", response_model=list[AlertOut])
def alerts(db: Session = Depends(get_db)):
    if not MODELS:
        return []
    ps = db.query(Project).filter(Project.status == "Ongoing").all()
    scored = portfolio_scores(ps)
    by_code = {x["project_code"]: x for x in scored}
    out = []
    for p in ps:
        x = by_code[p.project_code]
        if x["risk_score"] >= 70:
            signal = "Cost escalation + schedule risk" if x["cost_overrun_probability"] >= x["delay_probability"] else "Schedule delay risk"
            action = (
                "Cost-driver review and milestone recovery plan"
                if x["cost_overrun_probability"] >= x["delay_probability"]
                else "Milestone recovery plan and implementation review"
            )
            out.append({
                "project_code": p.project_code,
                "project_name": p.name,
                "severity": "CRITICAL" if x["risk_score"] >= 85 else "HIGH",
                "signal": signal,
                "probability": round(max(x["cost_overrun_probability"], x["delay_probability"]), 3),
                "recommended_action": action,
            })
    return sorted(out, key=lambda x: x["probability"], reverse=True)[:25]


@app.get("/api/analytics/summary")
def summary(db: Session = Depends(get_db)):
    ps = db.query(Project).filter(Project.status == "Ongoing").all()
    scored = portfolio_scores(ps)
    return {
        "projects": len(ps),
        "original_cost_cr": round(sum(p.original_cost_cr for p in ps), 2),
        "revised_cost_cr": round(sum(p.revised_cost_cr for p in ps), 2),
        "expenditure_cr": round(sum(p.expenditure_cr for p in ps), 2),
        "high_risk": sum(1 for x in scored if x["risk_level"] == "High"),
        "medium_risk": sum(1 for x in scored if x["risk_level"] == "Medium"),
        "low_risk": sum(1 for x in scored if x["risk_level"] == "Low"),
    }


@app.get("/api/analytics/sectors")
def sectors(db: Session = Depends(get_db)):
    ps = db.query(Project).filter(Project.status == "Ongoing").all()
    scored = portfolio_scores(ps)
    by_code = {x["project_code"]: x for x in scored}
    grouped = {}
    for p in ps:
        grouped.setdefault(p.sector, []).append(by_code[p.project_code]["risk_score"])
    data = [
        {
            "sector": sector,
            "projects": len(values),
            "avg_risk": round(sum(values) / len(values), 1),
            "high_risk": sum(1 for s in values if s >= 70),
        }
        for sector, values in grouped.items()
    ]
    return sorted(data, key=lambda x: x["avg_risk"], reverse=True)


@app.get("/api/models/metrics")
def model_metrics():
    path = MODEL_DIR / "metrics.json"
    return {
        "model_version": MODEL_VERSION,
        "metrics": json.loads(path.read_text()) if path.exists() else {},
    }


@app.post("/api/assistant", response_model=ChatResponse)
def assistant(req: ChatRequest, db: Session = Depends(get_db)):
    text = req.message.lower()
    evidence = []

    if req.project_code:
        p = db.query(Project).filter(Project.project_code == req.project_code).first()
        if p:
            pc, pdly, cost_pct, delay_m, score, level = (
                predict(MODELS, project_row(p))
                if MODELS else (0, 0, 0, 0, 0, "Unknown")
            )
            evidence = [
                f"{p.project_code}: physical progress {p.physical_progress_pct:.1f}%, schedule progress {p.schedule_progress_pct:.1f}%",
                f"Model: cost-risk {pc:.1%}, delay-risk {pdly:.1%}, PIRI {score:.1f} ({level})",
            ]
            if "why" in text or "risk" in text or "driver" in text:
                return {
                    "answer": (
                        f"{p.name} is currently classified as {level} risk. "
                        f"The model estimates {pc:.1%} cost-overrun probability and "
                        f"{pdly:.1%} delay probability. Review the Explainable AI "
                        f"drivers before taking action."
                    ),
                    "evidence": evidence,
                }

    return {
        "answer": (
            "I can answer from authorised project records and model outputs. "
            "Select a project and ask why it is risky, or ask about its cost and schedule risk. "
            "A production deployment should connect this endpoint to an approved LLM/RAG service with access controls."
        ),
        "evidence": evidence,
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
