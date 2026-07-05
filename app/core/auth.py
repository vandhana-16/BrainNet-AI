 
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db, Doctor

SECRET_KEY = "brainnet_secret_key_2024"
ALGORITHM  = "HS256"

pwd_ctx = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(p):
    return pwd_ctx.hash(p)
                                                                   #irun
def verify_password(plain, hashed):
    return pwd_ctx.verify(plain, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

def get_current_doctor(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        payload = decode_token(token)
        email   = payload.get("sub")
        if not email:
            raise HTTPException(401, "Invalid token")
        doc = db.query(Doctor).filter(Doctor.email == email).first()
        if not doc:
            raise HTTPException(401, "Doctor not found")
        return doc
    except JWTError:
        raise HTTPException(401, "Invalid token")
