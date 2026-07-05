
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session                               #irun
from app.database import get_db, Doctor
from app.core.auth import hash_password, verify_password, create_token
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["Auth"])

class RegisterReq(BaseModel):
    name: str
    email: str
    password: str
    hospital: str = ""

@router.post("/register")
def register(req: RegisterReq, db: Session = Depends(get_db)):
    if db.query(Doctor).filter(Doctor.email == req.email).first():
        raise HTTPException(400, "Email already registered")
    doc = Doctor(
        name=req.name, email=req.email,
        hashed_password=hash_password(req.password),
        hospital=req.hospital
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"message": "Registered successfully", "id": doc.id}

@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    doc = db.query(Doctor).filter(Doctor.email == form.username).first()
    if not doc or not verify_password(form.password, doc.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    token = create_token({"sub": doc.email})
    return {"access_token": token, "token_type": "bearer",
            "name": doc.name, "hospital": doc.hospital}
