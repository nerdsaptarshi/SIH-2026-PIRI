import os
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./piri.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    project_code = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(500), nullable=False)
    sector = Column(String(200), nullable=False)
    ministry = Column(String(300), nullable=False)
    original_cost_cr = Column(Float, default=0.0)
    revised_cost_cr = Column(Float, default=0.0)
    expenditure_cr = Column(Float, default=0.0)
    physical_progress_pct = Column(Float, default=0.0)
    schedule_progress_pct = Column(Float, default=0.0)
    planned_duration_months = Column(Float, default=0.0)
    elapsed_months = Column(Float, default=0.0)
    milestones_due = Column(Integer, default=0)
    milestones_delayed = Column(Integer, default=0)
    monthly_expenditure_growth_pct = Column(Float, default=0.0)
    cost_growth_pct = Column(Float, default=0.0)
    agency_delay_count = Column(Integer, default=0)
    contract_variation_count = Column(Integer, default=0)
    clearance_pending = Column(Boolean, default=False)
    last_update = Column(DateTime, nullable=True)
    status = Column(String(50), default="Ongoing", index=True)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    project_code = Column(String(100), index=True, nullable=False)
    cost_overrun_probability = Column(Float, nullable=False)
    delay_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    top_drivers = Column(Text, default="[]")
    model_version = Column(String(100), default="piri-rf-v1")
    created_at = Column(DateTime, nullable=False)

def init_db():
    Base.metadata.create_all(bind=engine)

def reset_demo_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
