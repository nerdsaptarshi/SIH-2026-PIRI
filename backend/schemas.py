from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class ProjectOut(BaseModel):
    model_config=ConfigDict(from_attributes=True)
    project_code:str; name:str; sector:str; ministry:str
    original_cost_cr:float; revised_cost_cr:float; expenditure_cr:float
    physical_progress_pct:float; schedule_progress_pct:float
    planned_duration_months:float; elapsed_months:float
    milestones_due:int; milestones_delayed:int
    monthly_expenditure_growth_pct:float; cost_growth_pct:float
    agency_delay_count:int; contract_variation_count:int
    clearance_pending:bool; last_update:Optional[datetime]=None; status:str

class PredictionOut(BaseModel):
    project_code:str; cost_overrun_probability:float; delay_probability:float
    risk_score:float; risk_level:str; top_drivers:list[dict[str,Any]]=Field(default_factory=list)
    model_version:str; cost_overrun_pct:float=0; predicted_delay_months:float=0

class AlertOut(BaseModel):
    project_code:str; project_name:str; severity:str; signal:str
    probability:float; recommended_action:str

class ChatRequest(BaseModel):
    message:str; project_code:Optional[str]=None

class ChatResponse(BaseModel):
    answer:str; evidence:list[str]=Field(default_factory=list)
