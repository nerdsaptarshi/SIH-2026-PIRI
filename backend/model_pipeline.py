from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "sector","ministry","original_cost_cr","revised_cost_cr","expenditure_cr",
    "physical_progress_pct","schedule_progress_pct","planned_duration_months",
    "elapsed_months","milestones_due","milestones_delayed",
    "monthly_expenditure_growth_pct","cost_growth_pct","agency_delay_count",
    "contract_variation_count","clearance_pending"
]
CAT = ["sector","ministry"]
NUM = [c for c in FEATURES if c not in CAT]
MODEL_VERSION = "piri-rf-v1"

def normalize_frame(data):
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    for col in FEATURES:
        if col not in frame.columns:
            frame[col] = 0
    frame = frame[FEATURES].copy()
    for col in CAT:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    for col in NUM:
        if col == "clearance_pending":
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    if frame["clearance_pending"].dtype == object:
        frame["clearance_pending"] = (
            frame["clearance_pending"].astype(str).str.lower()
            .isin(["true","1","yes","y"]).astype(int)
        )
    else:
        frame["clearance_pending"] = frame["clearance_pending"].fillna(0).astype(int)
    return frame

def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler(), NUM)
    ])

def build_models():
    return {
        "cost": Pipeline([("prep", preprocessor()), ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1))]),
        "delay": Pipeline([("prep", preprocessor()), ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=43, n_jobs=-1))]),
        "cost_pct": Pipeline([("prep", preprocessor()), ("model", RandomForestRegressor(
            n_estimators=300, random_state=44, n_jobs=-1))]),
        "delay_months": Pipeline([("prep", preprocessor()), ("model", RandomForestRegressor(
            n_estimators=300, random_state=45, n_jobs=-1))])
    }

def train_models(df, model_dir="models"):
    frame = normalize_frame(df)
    n = len(frame)
    split = max(1, min(n-1, int(n*0.8)))
    x_train, x_test = frame.iloc[:split], frame.iloc[split:]
    models = build_models()
    metrics = {}

    yc = df["cost_overrun"].astype(int)
    yd = df["schedule_delay"].astype(int)
    ycp = df["cost_overrun_pct"].astype(float)
    ydm = df["delay_months"].astype(float)

    models["cost"].fit(x_train, yc.iloc[:split])
    cp = models["cost"].predict_proba(x_test)[:,1]
    try: metrics["cost_auc"] = float(roc_auc_score(yc.iloc[split:], cp))
    except ValueError: metrics["cost_auc"] = None
    metrics["cost_accuracy"] = float(accuracy_score(yc.iloc[split:], (cp>=0.5).astype(int)))

    models["delay"].fit(x_train, yd.iloc[:split])
    dp = models["delay"].predict_proba(x_test)[:,1]
    try: metrics["delay_auc"] = float(roc_auc_score(yd.iloc[split:], dp))
    except ValueError: metrics["delay_auc"] = None
    metrics["delay_accuracy"] = float(accuracy_score(yd.iloc[split:], (dp>=0.5).astype(int)))

    models["cost_pct"].fit(x_train, ycp.iloc[:split])
    cpr = models["cost_pct"].predict(x_test)
    metrics["cost_mae_pct"] = float(mean_absolute_error(ycp.iloc[split:], cpr))
    metrics["cost_rmse_pct"] = float(np.sqrt(mean_squared_error(ycp.iloc[split:], cpr)))

    models["delay_months"].fit(x_train, ydm.iloc[:split])
    dmr = models["delay_months"].predict(x_test)
    metrics["delay_mae_months"] = float(mean_absolute_error(ydm.iloc[split:], dmr))
    metrics["delay_rmse_months"] = float(np.sqrt(mean_squared_error(ydm.iloc[split:], dmr)))

    out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, out/f"{name}.joblib")
    (out/"version.txt").write_text(MODEL_VERSION)
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2))
    return models, metrics

def load_models(model_dir="models"):
    d = Path(model_dir)
    files = {k:d/f"{k}.joblib" for k in ["cost","delay","cost_pct","delay_months"]}
    missing = [str(v) for v in files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("PIRI models missing. Run scripts/seed_and_train.py. Missing: "+str(missing))
    return {k:joblib.load(v) for k,v in files.items()}

def risk_level(score):
    return "High" if score >= 70 else ("Medium" if score >= 45 else "Low")

def predict(models, row):
    frame = normalize_frame(row)
    pc = float(models["cost"].predict_proba(frame)[:,1][0])
    pd = float(models["delay"].predict_proba(frame)[:,1][0])
    cost_pct = float(models["cost_pct"].predict(frame)[0])
    delay_months = float(models["delay_months"].predict(frame)[0])
    score = round(100*(0.45*pc + 0.40*pd + 0.15*min(max(cost_pct/30.0,0),1)),1)
    return round(pc,4), round(pd,4), round(cost_pct,2), round(delay_months,2), score, risk_level(score)

def batch_predict(models, data):
    frame = normalize_frame(data)
    cp = models["cost"].predict_proba(frame)[:,1]
    dp = models["delay"].predict_proba(frame)[:,1]
    cost_pct = models["cost_pct"].predict(frame)
    delay_months = models["delay_months"].predict(frame)
    scores = 100*(0.45*cp + 0.40*dp + 0.15*np.clip(cost_pct/30.0,0,1))
    return pd.DataFrame({
        "cost_overrun_probability": np.round(cp,4),
        "delay_probability": np.round(dp,4),
        "cost_overrun_pct": np.round(cost_pct,2),
        "delay_months": np.round(delay_months,2),
        "risk_score": np.round(scores,1),
        "risk_level": [risk_level(float(x)) for x in scores]
    })

def _feature_names(model):
    try: return list(model.named_steps["prep"].get_feature_names_out())
    except Exception: return FEATURES

def _shap_positive(model, frame):
    prep = model.named_steps["prep"]
    transformed = prep.transform(frame)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    values = explainer.shap_values(transformed)
    arr = np.asarray(values[1] if isinstance(values,list) and len(values)>1 else (values[0] if isinstance(values,list) else values))
    if arr.ndim == 3: arr = arr[0,:,1]
    elif arr.ndim == 2: arr = arr[0]
    else: arr = arr.reshape(-1)
    names = _feature_names(model)
    n=min(len(names),len(arr))
    pairs=[{"feature":names[i].replace("cat__","").replace("num__",""),"impact":round(float(arr[i]),4)} for i in range(n)]
    pairs.sort(key=lambda x:abs(x["impact"]),reverse=True)
    return pairs[:6]

def explain_project(models, row):
    frame=normalize_frame(row)
    result=[]
    try:
        result += [{"model":"Cost-overrun",**x} for x in _shap_positive(models["cost"],frame)]
    except Exception: pass
    try:
        result += [{"model":"Schedule-delay",**x} for x in _shap_positive(models["delay"],frame)]
    except Exception: pass
    return result
