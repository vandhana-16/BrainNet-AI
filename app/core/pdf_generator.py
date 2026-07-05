
import io
import cv2
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER
from PIL import Image as PILImage

def numpy_to_rl_image(arr, w_cm, h_cm):
    pil = PILImage.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=w_cm*cm, height=h_cm*cm)

def generate_pdf(result, metrics, output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm
    )
    C_WHITE  = colors.HexColor("#f1f5f9")
    C_GRAY   = colors.HexColor("#475569")
    C_DARK   = colors.HexColor("#0d0b1a")
    C_BORDER = colors.HexColor("#1e293b")
    C_CYAN   = colors.HexColor("#22d3ee")
    C_GREEN  = colors.HexColor("#34d399")
    C_YELLOW = colors.HexColor("#fbbf24")
    C_RED    = colors.HexColor("#f87171")

    tl          = result["trust_level"]
    trust_color = C_GREEN if tl=="HIGH" else C_YELLOW if tl=="MEDIUM" else C_RED

    def S(name, **kw): return ParagraphStyle(name, **kw)
    s_title   = S("t",  fontSize=22, textColor=C_WHITE,     fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)
    s_sub     = S("s",  fontSize=10, textColor=C_GRAY,      fontName="Helvetica",      alignment=TA_CENTER, spaceAfter=2)
    s_section = S("sc", fontSize=10, textColor=C_CYAN,      fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    s_body    = S("b",  fontSize=9,  textColor=C_WHITE,     fontName="Helvetica",      spaceAfter=4, leading=14)
    s_gray    = S("g",  fontSize=8,  textColor=C_GRAY,      fontName="Helvetica",      spaceAfter=3, leading=12)
    s_diag    = S("d",  fontSize=28, textColor=C_WHITE,     fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2)
    s_trust   = S("tr", fontSize=13, textColor=trust_color, fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2)
    s_footer  = S("f",  fontSize=7,  textColor=C_GRAY,      fontName="Helvetica",      alignment=TA_CENTER)

    def mrow(label, val, vc=None):
        vc = vc or C_WHITE
        return [
            Paragraph(f"<font color='#475569' size='8'>{label}</font>", s_body),
            Paragraph(f"<font color='#{vc.hexval()[2:]}' size='9'><b>{val}</b></font>", s_body),
        ]

    W     = A4[0] - 3.6*cm
    story = []

    story += [
        Spacer(1, 0.3*cm),
        Paragraph("BrainNet AI", s_title),
        Paragraph("Brain Tumor MRI Analysis Report", s_sub),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", s_sub),
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=0.5, color=C_BORDER),
    ]

    story.append(Paragraph("Primary Diagnosis", s_section))
    diag     = result["final_class"]
    is_tumor = diag.lower() != "notumor"
    rc       = C_RED if is_tumor else C_GREEN
    rl       = "HIGH RISK — Tumor Detected" if is_tumor else "LOW RISK — No Tumor Detected"

    story += [
        Paragraph(diag.upper(), s_diag),
        Paragraph(f"Trust Level: {tl}", s_trust),
        Paragraph(result["trust_msg"], s_gray),
        Spacer(1, 0.2*cm),
    ]
    rt = Table([[Paragraph(f"<font color='#{rc.hexval()[2:]}'><b>{rl}</b></font>", s_body)]], colWidths=[W])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),C_DARK), ("BOX",(0,0),(-1,-1),0.5,rc),
        ("TOPPADDING",(0,0),(-1,-1),10), ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),14), ("ALIGN",(0,0),(-1,-1),"CENTER"),
    ]))
    story += [rt, Spacer(1,0.2*cm), HRFlowable(width="100%",thickness=0.5,color=C_BORDER)]

    story.append(Paragraph("Spatial Consensus Confidence (SCC)", s_section))
    scc_table = Table([
        mrow("SCC Score",    str(result["scc_score"]),  trust_color),
        mrow("Trust Level",  tl,                        trust_color),
        mrow("Models Agree", "Yes" if result["models_agree"] else "No",
             C_GREEN if result["models_agree"] else C_RED),
        mrow("Decision",     result["trust_msg"]),
    ], colWidths=[W*0.45, W*0.55])
    scc_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),C_DARK), ("BOX",(0,0),(-1,-1),0.5,C_BORDER),
        ("LINEBELOW",(0,0),(-1,-2),0.3,C_BORDER),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),12), ("RIGHTPADDING",(0,0),(-1,-1),12),
    ]))
    story += [scc_table, Spacer(1,0.2*cm), HRFlowable(width="100%",thickness=0.5,color=C_BORDER)]

    story.append(Paragraph("Model Predictions", s_section))
    mt = Table([
        [Paragraph("<font color='#fbbf24' size='8'><b>EfficientNetB0</b></font>", s_body),
         Paragraph("<font color='#34d399' size='8'><b>ResNet50V2</b></font>", s_body)],
        [Paragraph(f"<font color='#f1f5f9' size='16'><b>{result['effnet_class'].upper()}</b></font>", s_body),
         Paragraph(f"<font color='#f1f5f9' size='16'><b>{result['resnet_class'].upper()}</b></font>", s_body)],
        [Paragraph(f"<font color='#94a3b8' size='8'>Confidence: {result['effnet_conf']}%</font>", s_body),
         Paragraph(f"<font color='#94a3b8' size='8'>Confidence: {result['resnet_conf']}%</font>", s_body)],
        [Paragraph(f"<font color='#64748b' size='8'>Area: {metrics['area_eff']}% | Lateral: {metrics['lateral_eff']}</font>", s_body),
         Paragraph(f"<font color='#64748b' size='8'>Area: {metrics['area_res']}% | Lateral: {metrics['lateral_res']}</font>", s_body)],
    ], colWidths=[W*0.5, W*0.5])
    mt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#0a0d1a")),
        ("BACKGROUND",(1,0),(1,-1),colors.HexColor("#0a0d1a")),
        ("BOX",(0,0),(0,-1),0.5,colors.HexColor("#fbbf2440")),
        ("BOX",(1,0),(1,-1),0.5,colors.HexColor("#34d39940")),
        ("LINEBELOW",(0,0),(-1,-2),0.3,C_BORDER),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),14), ("RIGHTPADDING",(0,0),(-1,-1),14),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story += [mt, Spacer(1,0.2*cm), HRFlowable(width="100%",thickness=0.5,color=C_BORDER)]

    story.append(Paragraph("GradCAM++ Activation Maps", s_section))
    iw = (W - 0.6*cm) / 4
    ih = iw
    it = Table([[
        numpy_to_rl_image(cv2.resize(result["img_orig"],(224,224)), iw/cm, ih/cm),
        numpy_to_rl_image(result["overlay_eff"],              iw/cm, ih/cm),
        numpy_to_rl_image(result["overlay_res"],              iw/cm, ih/cm),
        numpy_to_rl_image(metrics["consensus_overlay"],       iw/cm, ih/cm),
    ]], colWidths=[iw]*4)
    it.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),3),  ("RIGHTPADDING",(0,0),(-1,-1),3),
    ]))
    lb = Table([[
        Paragraph("<font color='#64748b' size='7'>Original MRI</font>",   s_gray),
        Paragraph("<font color='#fbbf24' size='7'>EfficientNetB0</font>", s_gray),
        Paragraph("<font color='#34d399' size='7'>ResNet50V2</font>",     s_gray),
        Paragraph("<font color='#22d3ee' size='7'>Consensus</font>",      s_gray),
    ]], colWidths=[iw]*4)
    lb.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),4)]))
    story += [it, lb, Spacer(1,0.2*cm), HRFlowable(width="100%",thickness=0.5,color=C_BORDER)]

    story.append(Paragraph("Tumor Region Metrics", s_section))
    tm = Table([
        mrow("Activation Area — EfficientNetB0", f"{metrics['area_eff']}%"),
        mrow("Activation Area — ResNet50V2",      f"{metrics['area_res']}%"),
        mrow("Activation Area — Consensus",       f"{metrics['area_consensus']}%"),
        mrow("Intensity — EfficientNetB0",        str(metrics['intensity_eff'])),
        mrow("Intensity — ResNet50V2",            str(metrics['intensity_res'])),
        mrow("Lateralization — Consensus",        metrics['lateral_consensus']),
        mrow("Sharpness Score",                   str(metrics['sharpness'])),
        mrow("Contrast Score",                    str(metrics['contrast'])),
        mrow("Image Quality",                     metrics['quality_flag'],
             C_GREEN if metrics['quality_flag']=='Good' else C_RED),
    ], colWidths=[W*0.55, W*0.45])
    tm.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),C_DARK), ("BOX",(0,0),(-1,-1),0.5,C_BORDER),
        ("LINEBELOW",(0,0),(-1,-2),0.3,C_BORDER),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),12), ("RIGHTPADDING",(0,0),(-1,-1),12),
    ]))
    story += [tm, Spacer(1,0.3*cm), HRFlowable(width="100%",thickness=0.5,color=C_BORDER)]

    story += [
        Spacer(1,0.3*cm),
        Paragraph("This report is generated by BrainNet AI for research purposes only. All findings must be verified by a qualified radiologist.", s_footer),
        Paragraph(f"BrainNet AI v2.0  ·  EfficientNetB0 + ResNet50V2  ·  SCC Trust Metric  ·  MySQL  ·  {datetime.now().strftime('%Y')}", s_footer),
    ]
    doc.build(story)
    return output_path
