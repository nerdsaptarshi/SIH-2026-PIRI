from datetime import datetime,timedelta
from pathlib import Path
import random,sys
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from backend.config import MODEL_DIR
from backend.db import Project,SessionLocal,reset_demo_db
from backend.model_pipeline import train_models

random.seed(42); np.random.seed(42)
SECTORS=["Roads & Highways","Railways","Power","Water Resources","Urban Development","Petroleum & Natural Gas"]
MINISTRIES={
"Roads & Highways":"Ministry of Road Transport & Highways","Railways":"Ministry of Railways",
"Power":"Ministry of Power","Water Resources":"Ministry of Jal Shakti",
"Urban Development":"Ministry of Housing & Urban Affairs",
"Petroleum & Natural Gas":"Ministry of Petroleum & Natural Gas"}

def make_projects(n=120):
    rows=[]
    for i in range(1,n+1):
        sector=random.choice(SECTORS); original=round(random.uniform(180,8000),2)
        physical=round(random.uniform(8,96),1); planned=random.randint(18,84)
        elapsed=round(planned*random.uniform(.35,1.25),1)
        schedule=max(2,min(100,physical+np.random.normal(0,8)-max(0,elapsed/planned-.85)*25))
        due=max(1,int(planned/6))
        delayed=min(due,max(0,int(np.random.poisson(1+max(0,(elapsed/planned-.75)*8)+(100-schedule)/35))))
        agency=max(0,int(np.random.poisson(.7+delayed*.35)))
        variations=max(0,int(np.random.poisson(.5+original/5000)))
        clearance=random.random() < (.08+.18*(delayed>=2))
        monthly=round(np.random.normal(2.5+delayed*1.3,3.5),2)
        growth=round(max(-1,np.random.normal(4+delayed*2.2+variations*1.6+(8 if clearance else 0),5)),2)
        revised=round(original*(1+max(growth,0)/100),2)
        expenditure=round(min(revised*.98,original*max(.03,min(.98,physical/100+np.random.normal(0,.05)))),2)
        cost_signal=growth+variations*2+monthly*.7+(7 if clearance else 0)+np.random.normal(0,3)
        delay_signal=(100-schedule)+delayed*9+agency*5+max(0,elapsed/planned-1)*35+(8 if clearance else 0)+np.random.normal(0,5)
        co=int(cost_signal>=13); sd=int(delay_signal>=42)
        cost_pct=round(max(0,growth+np.random.normal(0,2.5)+co*3.5),2)
        delay_m=round(max(0,(delay_signal-18)/9+np.random.normal(0,1.2)+sd*1.5),2)
        rows.append(dict(
            project_code=f"P-{1000+i}",name=f"{sector} Infrastructure Project {i:03d}",
            sector=sector,ministry=MINISTRIES[sector],original_cost_cr=original,
            revised_cost_cr=revised,expenditure_cr=expenditure,
            physical_progress_pct=round(float(physical),1),schedule_progress_pct=round(float(schedule),1),
            planned_duration_months=float(planned),elapsed_months=float(elapsed),
            milestones_due=due,milestones_delayed=delayed,
            monthly_expenditure_growth_pct=monthly,cost_growth_pct=growth,
            agency_delay_count=agency,contract_variation_count=variations,
            clearance_pending=clearance,last_update=datetime.utcnow()-timedelta(days=random.randint(0,90)),
            status="Ongoing",cost_overrun=co,schedule_delay=sd,cost_overrun_pct=cost_pct,delay_months=delay_m))
    return pd.DataFrame(rows)

def main():
    df=make_projects(120)
    reset_demo_db()
    db=SessionLocal()
    try:
        for r in df.to_dict("records"):
            fields=["project_code","name","sector","ministry","original_cost_cr","revised_cost_cr",
            "expenditure_cr","physical_progress_pct","schedule_progress_pct","planned_duration_months",
            "elapsed_months","milestones_due","milestones_delayed","monthly_expenditure_growth_pct",
            "cost_growth_pct","agency_delay_count","contract_variation_count","clearance_pending",
            "last_update","status"]
            db.add(Project(**{k:r[k] for k in fields}))
        db.commit()
    finally: db.close()
    _,metrics=train_models(df,MODEL_DIR)
    print(f"PIRI bootstrap complete: {len(df)} synthetic projects")
    print(metrics)

if __name__=="__main__": main()
