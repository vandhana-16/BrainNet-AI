
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db, Scan
from app.core.auth import get_current_doctor, decode_token
from app.core.ml_engine import run_inference, numpy_to_b64
from app.core.pdf_generator import generate_pdf
import numpy as np
import cv2, uuid, os
from datetime import datetime

router = APIRouter(prefix="/scan", tags=["Scan"])

UPLOAD_DIR = "/content/drive/MyDrive/brain_tumor_project/uploads"
REPORT_DIR = "/content/drive/MyDrive/brain_tumor_project/reports"

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form("Anonymous"),
    patient_age: int = Form(None),
    db: Session = Depends(get_db),
    doctor = Depends(get_current_doctor)
):
    uid      = str(uuid.uuid4())[:8]
    filepath = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    img_array = np.frombuffer(contents, np.uint8)
    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result    = run_inference(img_rgb)
    metrics   = result["metrics"]

    scan = Scan(
        doctor_id        = doctor.id,
        patient_name     = patient_name,
        patient_age      = patient_age,
        scan_path        = filepath,
        effnet_class     = result["effnet_class"],
        effnet_conf      = result["effnet_conf"],
        resnet_class     = result["resnet_class"],
        resnet_conf      = result["resnet_conf"],
        final_class      = result["final_class"],
        models_agree     = result["models_agree"],
        scc_score        = result["scc_score"],
        trust_level      = result["trust_level"],
        trust_msg        = result["trust_msg"],
        decision         = result["decision"],
        area_eff         = metrics["area_eff"],
        area_res         = metrics["area_res"],
        area_consensus   = metrics["area_consensus"],
        intensity_eff    = metrics["intensity_eff"],
        intensity_res    = metrics["intensity_res"],
        lateral_eff      = metrics["lateral_eff"],
        lateral_res      = metrics["lateral_res"],
        lateral_consensus= metrics["lateral_consensus"],
        sharpness        = metrics["sharpness"],
        contrast         = metrics["contrast"],
        quality_flag     = metrics["quality_flag"],
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "scan_id"      : scan.id,
        "effnet_class" : result["effnet_class"],
        "effnet_conf"  : result["effnet_conf"],
        "resnet_class" : result["resnet_class"],
        "resnet_conf"  : result["resnet_conf"],
        "final_class"  : result["final_class"],
        "models_agree" : result["models_agree"],
        "scc_score"    : result["scc_score"],
        "trust_level"  : result["trust_level"],
        "trust_msg"    : result["trust_msg"],
        "decision"     : result["decision"],
        "eff_probs"    : result["eff_probs"],
        "res_probs"    : result["res_probs"],
        "original"     : numpy_to_b64(cv2.resize(img_rgb,(224,224))),
        "overlay_eff"  : numpy_to_b64(result["overlay_eff"]),
        "overlay_res"  : numpy_to_b64(result["overlay_res"]),
        "consensus"    : numpy_to_b64(metrics["consensus_overlay"]),
        "area_eff"     : metrics["area_eff"],
        "area_res"     : metrics["area_res"],
        "area_consensus": metrics["area_consensus"],
        "intensity_eff": metrics["intensity_eff"],
        "intensity_res": metrics["intensity_res"],
        "lateral_con"  : metrics["lateral_consensus"],
        "lateral_eff"  : metrics["lateral_eff"],
        "lateral_res"  : metrics["lateral_res"],
        "sharpness"    : metrics["sharpness"],
        "contrast"     : metrics["contrast"],
        "quality_flag" : metrics["quality_flag"],
    }


@router.get("/download/{scan_id}")
def download_report(
    scan_id: int,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    # Decode token manually since window.open cant send headers
    try:
        payload = decode_token(token)
        email   = payload.get("sub")
        if not email:
            raise HTTPException(401, "Invalid token")
        from app.database import Doctor
        doctor = db.query(Doctor).filter(Doctor.email == email).first()
        if not doctor:
            raise HTTPException(401, "Doctor not found")
    except Exception:
        raise HTTPException(401, "Invalid token")

    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.doctor_id == doctor.id).first()
    if not scan:
        raise HTTPException(404, "Scan not found")

    img_orig = cv2.cvtColor(cv2.imread(scan.scan_path), cv2.COLOR_BGR2RGB)

    result = {
        "final_class"  : scan.final_class,
        "effnet_class" : scan.effnet_class,
        "effnet_conf"  : scan.effnet_conf,
        "resnet_class" : scan.resnet_class,
        "resnet_conf"  : scan.resnet_conf,
        "models_agree" : scan.models_agree,
        "scc_score"    : scan.scc_score,
        "trust_level"  : scan.trust_level,
        "trust_msg"    : scan.trust_msg,
        "img_orig"     : img_orig,
        "overlay_eff"  : img_orig,
        "overlay_res"  : img_orig,
    }
    metrics = {
        "area_eff"         : scan.area_eff,
        "area_res"         : scan.area_res,
        "area_consensus"   : scan.area_consensus,
        "intensity_eff"    : scan.intensity_eff,
        "intensity_res"    : scan.intensity_res,
        "lateral_eff"      : scan.lateral_eff,
        "lateral_res"      : scan.lateral_res,
        "lateral_consensus": scan.lateral_consensus,
        "sharpness"        : scan.sharpness,
        "contrast"         : scan.contrast,
        "quality_flag"     : scan.quality_flag,
        "consensus_overlay": img_orig,
    }

    pdf_path = os.path.join(REPORT_DIR, f"report_{scan_id}.pdf")
    generate_pdf(result, metrics, pdf_path)
    return FileResponse(
        pdf_path,
        filename=f"BrainNet_Report_{scan_id}.pdf",
        media_type="application/pdf"
    )


@router.get("/history")
def history(db: Session = Depends(get_db), doctor = Depends(get_current_doctor)):
    scans = db.query(Scan).filter(Scan.doctor_id == doctor.id).order_by(Scan.created_at.desc()).all()
    return [{
        "id"          : s.id,
        "patient_name": s.patient_name,
        "final_class" : s.final_class,
        "scc_score"   : s.scc_score,
        "trust_level" : s.trust_level,
        "decision"    : s.decision,
        "created_at"  : str(s.created_at),
    } for s in scans]
