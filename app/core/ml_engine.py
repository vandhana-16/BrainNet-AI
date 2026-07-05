
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from PIL import Image as PILImage
from io import BytesIO
import base64

SAVE_DIR     = "/content/drive/MyDrive/brain_tumor_project"
CLASSES      = ["glioma", "meningioma", "notumor", "pituitary"]
EFFNET_LAYER = "top_conv"
RESNET_LAYER = "conv5_block3_3_conv"

print("Loading models...")
effnet_model = tf.keras.models.load_model(f"{SAVE_DIR}/EfficientNetB0_final.keras")
resnet_model = tf.keras.models.load_model(f"{SAVE_DIR}/ResNet50V2_final.keras")

def get_base(model):
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 5:
            return layer
    return None

effnet_base = get_base(effnet_model)
resnet_base = get_base(resnet_model)
print("Models loaded.")

def preprocess(img_array, preprocess_fn):
    img = cv2.resize(img_array, (224, 224))
    arr = np.expand_dims(img.copy(), axis=0).astype(np.float32)
    return img, preprocess_fn(arr)

def generate_gradcam(base_model, arr, class_idx, layer_name):
    target     = base_model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[target.output, base_model.output]
    )
    img_tensor = tf.cast(arr, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = grad_model(img_tensor, training=False)
        tape.watch(conv_out)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return np.zeros((7, 7))
    pooled = tf.reduce_mean(grads, axis=[0, 1, 2])
    cam    = conv_out[0] @ pooled[..., tf.newaxis]
    cam    = tf.squeeze(cam)
    cam    = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam

def overlay_heatmap(img_rgb, cam, alpha=0.4):
    cam_r   = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (heatmap * alpha + img_rgb * (1 - alpha)).astype(np.uint8)

def compute_scc(cam1, cam2, threshold=0.1):
    m1    = (cv2.resize(cam1, (224,224)) >= threshold).astype(np.float32)
    m2    = (cv2.resize(cam2, (224,224)) >= threshold).astype(np.float32)
    inter = np.sum(m1 * m2)
    union = np.sum(np.clip(m1 + m2, 0, 1))
    return float(inter / union) if union > 0 else 0.0

def get_trust(scc, agree):
    if not agree:
        return "LOW", "Models disagree — radiologist review recommended", "refer"
    if scc >= 0.35:
        return "HIGH", "High spatial consensus — prediction is trustworthy", "predict"
    elif scc >= 0.15:
        return "MEDIUM", "Moderate spatial consensus — interpret with caution", "caution"
    return "LOW", "Low spatial consensus — radiologist review recommended", "refer"

def compute_tumor_metrics(cam_eff, cam_res, img_orig, threshold=0.1):
    S         = 224
    cam_e     = cv2.resize(cam_eff, (S, S))
    cam_r     = cv2.resize(cam_res, (S, S))
    consensus = (cam_e + cam_r) / 2.0
    mask_e    = (cam_e    >= threshold).astype(np.uint8)
    mask_r    = (cam_r    >= threshold).astype(np.uint8)
    mask_con  = (consensus >= threshold).astype(np.uint8)
    total     = S * S

    def get_centroid(mask):
        M = cv2.moments(mask)
        if M["m00"] == 0: return (S//2, S//2)
        return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

    def get_lateral(c):
        if c[0] < S*0.4:  return "Left hemisphere"
        if c[0] > S*0.6:  return "Right hemisphere"
        return "Central / midline"

    gray      = cv2.cvtColor(cv2.resize(img_orig,(S,S)), cv2.COLOR_RGB2GRAY)
    sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    contrast  = round(float(gray.std()), 2)

    if sharpness < 100:  quality_flag = "Low — image may be blurry"
    elif contrast < 30:  quality_flag = "Low — insufficient contrast"
    else:                quality_flag = "Good"

    con_norm    = (consensus * 255).astype(np.uint8)
    con_color   = cv2.cvtColor(cv2.applyColorMap(con_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    con_overlay = (con_color * 0.4 + cv2.resize(img_orig,(S,S)) * 0.6).astype(np.uint8)
    cc          = get_centroid(mask_con)

    return {
        "area_eff"          : round(mask_e.sum()/total*100, 2),
        "area_res"          : round(mask_r.sum()/total*100, 2),
        "area_consensus"    : round(mask_con.sum()/total*100, 2),
        "intensity_eff"     : round(float(cam_e[mask_e==1].mean()),4) if mask_e.sum()>0 else 0.0,
        "intensity_res"     : round(float(cam_r[mask_r==1].mean()),4) if mask_r.sum()>0 else 0.0,
        "lateral_eff"       : get_lateral(get_centroid(mask_e)),
        "lateral_res"       : get_lateral(get_centroid(mask_r)),
        "lateral_consensus" : get_lateral(cc),
        "sharpness"         : sharpness,
        "contrast"          : contrast,
        "quality_flag"      : quality_flag,
        "consensus_overlay" : con_overlay,
    }

def numpy_to_b64(arr):
    buf = BytesIO()
    PILImage.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_inference(img_rgb):
    img_eff, arr_eff = preprocess(img_rgb, effnet_preprocess)
    img_res, arr_res = preprocess(img_rgb, resnet_preprocess)
    pred_eff         = effnet_model.predict(arr_eff, verbose=0)[0]
    pred_res         = resnet_model.predict(arr_res, verbose=0)[0]
    class_eff        = int(np.argmax(pred_eff))
    class_res        = int(np.argmax(pred_res))
    conf_eff         = float(pred_eff[class_eff])
    conf_res         = float(pred_res[class_res])
    cam_eff          = generate_gradcam(effnet_base, arr_eff, class_eff, EFFNET_LAYER)
    cam_res          = generate_gradcam(resnet_base, arr_res, class_res, RESNET_LAYER)
    scc_score        = compute_scc(cam_eff, cam_res)
    models_agree     = (class_eff == class_res)
    trust, msg, dec  = get_trust(scc_score, models_agree)
    metrics          = compute_tumor_metrics(cam_eff, cam_res, img_rgb)
    ov_eff           = overlay_heatmap(img_eff, cam_eff)
    ov_res           = overlay_heatmap(img_res, cam_res)

    return {
        "effnet_class" : CLASSES[class_eff],
        "effnet_conf"  : round(conf_eff*100, 1),
        "resnet_class" : CLASSES[class_res],
        "resnet_conf"  : round(conf_res*100, 1),
        "final_class"  : CLASSES[class_eff] if models_agree else "uncertain",
        "models_agree" : models_agree,
        "scc_score"    : round(scc_score, 4),
        "trust_level"  : trust,
        "trust_msg"    : msg,
        "decision"     : dec,
        "eff_probs"    : {CLASSES[i]: round(float(pred_eff[i])*100,1) for i in range(4)},
        "res_probs"    : {CLASSES[i]: round(float(pred_res[i])*100,1) for i in range(4)},
        "overlay_eff"  : ov_eff,
        "overlay_res"  : ov_res,
        "img_orig"     : img_rgb,
        "metrics"      : metrics,
    }
