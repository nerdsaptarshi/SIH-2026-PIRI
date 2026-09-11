from datetime import datetime
from pathlib import Path
import json
import math

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .config import MODEL_DIR
from .db import Project, Prediction, get_db, init_db
from .model_pipeline import FEATURES, MODEL_VERSION, explain_project, load_models, predict, batch_predict
from .schemas import AlertOut, ChatRequest, ChatResponse, PredictionOut, ProjectOut

ROOT=Path(__file__).resolve().parent.parent
FRONTEND=ROOT/"frontend"
app=FastAPI(title="PIRI — Predictive Infrastructure Risk Intelligence",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
_models=None

@app.on_event("startup")
def startup():
    global _models
    init_db()
    _models=load_models(MODEL_DIR)

def models():
    if _models is None: raise HTTPException(503,"AI models are not loaded")
    return _models

def project_dict(p):
    return {c.name:getattr(p,c.name) for c in Project.__table__.columns if c.name!="id"}

def portfolio_scores(projects):
    if not projects: return pd.DataFrame()
    df=pd.DataFrame([project_dict(p) for p in projects])
    scores=batch_predict(models(),df)
    return pd.concat([df.reset_index(drop=True),scores.reset_index(drop=True)],axis=1)

def alert_from(row):
    cost=float(row.cost_overrun_probability); delay=float(row.delay_probability)
    if row.risk_level=="High":
        severity="CRITICAL" if float(row.risk_score)>=85 else "HIGH"
        if cost>=delay:
            signal="High probability of cost overrun"
            action="Review cost growth, contract variations and expenditure trajectory; assign corrective action."
            prob=cost
        else:
            signal="High probability of schedule delay"
            action="Review delayed milestones, agency constraints and pending clearances; escalate recovery plan."
            prob=delay
        return dict(project_code=row.project_code,project_name=row.name,severity=severity,signal=signal,
                    probability=round(prob,4),recommended_action=action)
    return None

@app.get("/health")
def health(): return {"status":"ok","model_version":MODEL_VERSION}

@app.get("/")
def root(): return FileResponse(FRONTEND/"index.html")

@app.get("/api/projects",response_model=list[ProjectOut])
def projects(q: str|None=None, sector: str|None=None, risk: str|None=None, db: Session=Depends(get_db)):
    ps=db.query(Project).filter(Project.status=="Ongoing").order_by(Project.project_code).all()
    if q: ps=[p for p in ps if q.lower() in (p.name+" "+p.project_code+" "+p.ministry).lower()]
    if sector: ps=[p for p in ps if p.sector==sector]
    if risk:
        scored=portfolio_scores(ps); ps=[p for p in ps if str(scored.loc[scored.project_code==p.project_code,"risk_level"].iloc[0]).lower()==risk.lower()]
    return ps

@app.get("/api/projects/{code}/prediction",response_model=PredictionOut)
def project_prediction(code:str,db:Session=Depends(get_db)):
    p=db.query(Project).filter(Project.project_code==code).first()
    if not p: raise HTTPException(404,"Project not found")
    row=pd.DataFrame([project_dict(p)])
    pc,pd,cp,dm,score,level=predict(models(),row)
    drivers=explain_project(models(),row)
    pred=Prediction(project_code=code,cost_overrun_probability=pc,delay_probability=pd,
                    risk_score=score,risk_level=level,top_drivers=json.dumps(drivers),
                    model_version=MODEL_VERSION,created_at=datetime.utcnow())
    db.add(pred); db.commit()
    return PredictionOut(project_code=code,cost_overrun_probability=pc,delay_probability=pd,
                         risk_score=score,risk_level=level,top_drivers=drivers,
                         model_version=MODEL_VERSION,cost_overrun_pct=cp,predicted_delay_months=dm)

@app.post("/api/projects/{code}/predict",response_model=PredictionOut)
def run_prediction(code:str,db:Session=Depends(get_db)):
    return project_prediction(code,db)

@app.get("/api/alerts",response_model=list[AlertOut])
def alerts(db:Session=Depends(get_db)):
    scored=portfolio_scores(db.query(Project).filter(Project.status=="Ongoing").all())
    if scored.empty:return []
    out=[alert_from(r) for r in scored.itertuples()]
    return [x for x in out if x][:12]

@app.get("/api/analytics/summary")
def summary(db:Session=Depends(get_db)):
    scored=portfolio_scores(db.query(Project).filter(Project.status=="Ongoing").all())
    if scored.empty:return {"ongoing_projects":0,"original_cost_cr":0,"revised_cost_cr":0,"expenditure_cr":0,"high_risk":0,"medium_risk":0,"low_risk":0,"avg_risk":0}
    return {"ongoing_projects":len(scored),"original_cost_cr":round(scored.original_cost_cr.sum(),2),
            "revised_cost_cr":round(scored.revised_cost_cr.sum(),2),"expenditure_cr":round(scored.expenditure_cr.sum(),2),
            "high_risk":int((scored.risk_level=="High").sum()),"medium_risk":int((scored.risk_level=="Medium").sum()),
            "low_risk":int((scored.risk_level=="Low").sum()),"avg_risk":round(scored.risk_score.mean(),1)}

@app.get("/api/analytics/sectors")
def sectors(db:Session=Depends(get_db)):
    s=portfolio_scores(db.query(Project).filter(Project.status=="Ongoing").all())
    if s.empty:return []
    g=s.groupby("sector")
    return [{"sector":name,"projects":int(len(grp)),"avg_risk":round(float(grp.risk_score.mean()),1),
             "high_risk":int((grp.risk_level=="High").sum()),"cost_risk":round(float(grp.cost_overrun_probability.mean()),3),
             "delay_risk":round(float(grp.delay_probability.mean()),3)} for name,grp in sorted(g)]

@app.get("/api/models/metrics")
def model_metrics():
    path=MODEL_DIR/"metrics.json"
    return json.loads(path.read_text()) if path.exists() else {"model_version":MODEL_VERSION}

@app.get("/api/dashboard")
def dashboard(db:Session=Depends(get_db)):
    ps=db.query(Project).filter(Project.status=="Ongoing").order_by(Project.project_code).all()
    s=portfolio_scores(ps)
    if s.empty:
        return {"model_version":MODEL_VERSION,"generated_at":datetime.utcnow().isoformat()+"Z",
                "summary":{},"sectors":[],"alerts":[],"projects":[]}
    summary={"ongoing_projects":len(s),"original_cost_cr":round(s.original_cost_cr.sum(),2),
             "revised_cost_cr":round(s.revised_cost_cr.sum(),2),"expenditure_cr":round(s.expenditure_cr.sum(),2),
             "high_risk":int((s.risk_level=="High").sum()),"medium_risk":int((s.risk_level=="Medium").sum()),
             "low_risk":int((s.risk_level=="Low").sum()),"avg_risk":round(float(s.risk_score.mean()),1)}
    sg=s.groupby("sector")
    sector_data=[{"sector":n,"projects":int(len(g)),"avg_risk":round(float(g.risk_score.mean()),1),
                  "high_risk":int((g.risk_level=="High").sum()),
                  "cost_risk":round(float(g.cost_overrun_probability.mean()),3),
                  "delay_risk":round(float(g.delay_probability.mean()),3)} for n,g in sorted(sg)]
    alerts_data=[x for x in [alert_from(r) for r in s.itertuples()] if x]
    alerts_data=sorted(alerts_data,key=lambda x:(0 if x["severity"]=="CRITICAL" else 1,-x["probability"]))[:12]
    projects=[]
    for r in s.itertuples():
        projects.append({k:(None if (isinstance(getattr(r,k),float) and math.isnan(getattr(r,k))) else getattr(r,k))
                         for k in ["project_code","name","sector","ministry","original_cost_cr","revised_cost_cr","expenditure_cr",
                         "physical_progress_pct","schedule_progress_pct","planned_duration_months","elapsed_months",
                         "milestones_due","milestones_delayed","monthly_expenditure_growth_pct","cost_growth_pct",
                         "agency_delay_count","contract_variation_count","clearance_pending","last_update","status",
                         "cost_overrun_probability","delay_probability","cost_overrun_pct","delay_months","risk_score","risk_level"]})
    return {"model_version":MODEL_VERSION,"generated_at":datetime.utcnow().isoformat()+"Z",
            "summary":summary,"sectors":sector_data,"alerts":alerts_data,"projects":projects}

@app.post("/api/assistant",response_model=ChatResponse)
def assistant(req:ChatRequest,db:Session=Depends(get_db)):
    p=None
    if req.project_code: p=db.query(Project).filter(Project.project_code==req.project_code).first()
    if not p and req.project_code: raise HTTPException(404,"Project not found")
    text=req.message.lower()
    if p:
        pc,pd,cp,dm,score,level=predict(models(),pd.DataFrame([project_dict(p)]))
        if "why" in text or "risk" in text:
            answer=(f"{p.name} is currently rated {level} with a PIRI score of {score}. "
                    f"Estimated cost-overrun risk is {pc:.0%}, schedule-delay risk is {pd:.0%}. "
                    f"Predicted cost overrun is about {cp:.1f}% and expected delay is about {dm:.1f} months. "
                    "Use the Explainable AI panel to review the strongest SHAP drivers.")
        elif "cost" in text:
            answer=f"Cost-overrun risk is {pc:.0%}; the model estimates about {cp:.1f}% cost growth for this project."
        elif "delay" in text or "schedule" in text:
            answer=f"Schedule-delay risk is {pd:.0%}; the model estimates about {dm:.1f} months of delay."
        else:
            answer=f"{p.name}: PIRI {score} ({level}), cost risk {pc:.0%}, delay risk {pd:.0%}. Ask 'why is this risky?', 'cost', or 'delay' for a focused answer."
        return {"answer":answer,"evidence":["Current project record","PIRI Random Forest prediction","SHAP explainability layer"]}
    return {"answer":"Select a project first, then ask me about its risk, cost, delay, or drivers.","evidence":["PIRI project portfolio"]}

app.mount("/static",StaticFiles(directory=FRONTEND),name="static")
