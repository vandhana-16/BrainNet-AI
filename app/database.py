
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Save database to Drive — persists across sessions
DB_PATH      = "/content/drive/MyDrive/brain_tumor_project/brainnet.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

class Doctor(Base):
    __tablename__ = "doctors"
    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(100), nullable=False)
    email           = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    hospital        = Column(String(150), default="")
    created_at      = Column(DateTime, default=datetime.utcnow)

class Scan(Base):
    __tablename__ = "scans"
    id                = Column(Integer, primary_key=True, index=True)
    doctor_id         = Column(Integer, nullable=False)
    patient_name      = Column(String(100), default="Anonymous")
    patient_age       = Column(Integer, nullable=True)
    scan_path         = Column(String(300))
    report_path       = Column(String(300), nullable=True)
    effnet_class      = Column(String(50))
    effnet_conf       = Column(Float)
    resnet_class      = Column(String(50))
    resnet_conf       = Column(Float)
    final_class       = Column(String(50))
    models_agree      = Column(Boolean)
    scc_score         = Column(Float)
    trust_level       = Column(String(20))
    trust_msg         = Column(String(300))
    decision          = Column(String(20))
    area_eff          = Column(Float)
    area_res          = Column(Float)
    area_consensus    = Column(Float)
    intensity_eff     = Column(Float)
    intensity_res     = Column(Float)
    lateral_eff       = Column(String(50))
    lateral_res       = Column(String(50))
    lateral_consensus = Column(String(50))
    sharpness         = Column(Float)
    contrast          = Column(Float)
    quality_flag      = Column(String(50))
    created_at        = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Tables ready.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
