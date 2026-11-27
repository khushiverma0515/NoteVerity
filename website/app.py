import io
import os
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO

# --------------------------------------------------
# PATHS & CONFIG
# --------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_DIR      = BASE_DIR / "model"
MIN_DENOM_CONF  = 60.0   # minimum % to count as a valid detection
MIN_BOX_AREA_PC = 5.0    # note must cover ≥5% of frame
# Indian notes are ~1.9–2.6× wider than tall; allow 1.3–4.0 for perspective distortion
NOTE_AR_MIN     = 1.3
NOTE_AR_MAX     = 4.0

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NoteVerity", page_icon="💵", layout="wide")

# --------------------------------------------------
# LOAD MODELS  (cached — loads only once)
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models… first run takes ~30 seconds.")
def load_models():
    import gc
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    yolo = YOLO(str(MODEL_DIR / "best.pt"))
    yolo.fuse()

    resnet = resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(
        torch.load(MODEL_DIR / "real_fake_model.pth", map_location="cpu", weights_only=True)
    )
    resnet.eval()
    gc.collect()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return yolo, resnet, tfm

denomination_model, authenticity_model, transform = load_models()

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
def run_inference(image_bytes: bytes) -> dict:
    """
    Returns a dict with keys:
      status  : "no_note" | "unclear" | "detected" | "error"
      denomination_detections : list of {denomination, confidence}
      authenticity, authenticity_confidence
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"status": "error", "denomination_detections": [],
                "authenticity": "UNCERTAIN", "authenticity_confidence": 0.0}

    img_w, img_h = image.width, image.height
    img_area = img_w * img_h

    # Use low YOLO threshold so we see ALL candidates and decide ourselves
    with torch.inference_mode():
        results = denomination_model.predict(
            image, conf=0.25, verbose=False, imgsz=640, device="cpu"
        )

    detections   = []   # passed all filters → genuine note
    unclear_seen = False  # something note-like found but not confident / too far

    for result in results:
        for box in result.boxes:
            conf_pct = round(float(box.conf.item()) * 100, 2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            box_area_pc  = (w * h / img_area) * 100
            aspect_ratio = w / max(h, 1)

            # ── SHAPE FILTER ──────────────────────────────────────────────
            # Indian notes are ~1.9–2.6:1 wide; allow generous range for tilt
            if not (NOTE_AR_MIN <= aspect_ratio <= NOTE_AR_MAX):
                continue   # wrong shape → not a note (catches credit cards held straight)

            # ── SIZE FILTER ───────────────────────────────────────────────
            if box_area_pc < MIN_BOX_AREA_PC:
                unclear_seen = True   # note-shaped but too small / far away
                continue

            # ── CONFIDENCE FILTER ─────────────────────────────────────────
            if conf_pct < MIN_DENOM_CONF:
                unclear_seen = True   # note-shaped, right size, but model unsure
                continue

            class_id = int(box.cls.item())
            detections.append({
                "denomination": str(result.names[class_id]),
                "confidence":   conf_pct,
            })

    # ── NOTHING PASSED ALL FILTERS ────────────────────────────────────────
    if not detections:
        status = "unclear" if unclear_seen else "no_note"
        return {"status": status, "denomination_detections": [],
                "authenticity": "UNCERTAIN", "authenticity_confidence": 0.0}

    # ── RUN AUTHENTICITY ON CONFIRMED NOTE ────────────────────────────────
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs   = torch.softmax(authenticity_model(tensor), dim=1)[0]
    classes = ["fake", "real"]
    pred_id = int(torch.argmax(probs).item())

    return {
        "status":                  "detected",
        "denomination_detections": detections,
        "authenticity":            classes[pred_id].upper(),
        "authenticity_confidence": round(float(probs[pred_id]) * 100, 2),
    }

# --------------------------------------------------
# PAGE CSS + HERO
# --------------------------------------------------
st.html("""
<style>
:root { --silver:#D9C4B9; --tuscany:#B49E94; }
.stApp {
    background:
        radial-gradient(ellipse 70% 45% at 50% -8%, rgba(114,70,59,.42), transparent 70%),
        linear-gradient(180deg, #221512 0%, #1b100d 100%);
}
[data-testid="stHeader"] { background:transparent; }
#MainMenu { visibility:hidden; }
.block-container { max-width:1180px; padding-top:2rem; padding-bottom:40px; }
.hero       { text-align:center; padding:24px 0 44px; }
.eyebrow    { color:var(--tuscany); font-size:10px; letter-spacing:3px; opacity:.75; }
.hero-title { color:var(--silver); font-size:clamp(38px,6vw,58px); letter-spacing:-1.8px; font-weight:760; margin:14px 0; }
.hero-copy  { color:var(--tuscany); font-size:15px; opacity:.85; }
.site-footer{ margin-top:30px; padding:20px; border-top:1px solid rgba(217,196,185,.1);
              text-align:center; color:var(--tuscany); opacity:.6; font-size:11px; }
</style>
<div class="hero">
    <div class="eyebrow">NOTEVERITY &middot; CURRENCY VERIFICATION</div>
    <div class="hero-title">NoteVerity</div>
    <div class="hero-copy">AI-powered currency denomination detection and<br>
        authenticity screening through your webcam.</div>
</div>
""")


# --------------------------------------------------
# RESULT RENDERER
# --------------------------------------------------
def show_result(result: dict, slot) -> None:
    dets = result.get("denomination_detections", [])
    if not dets:
        slot.warning("⚠️ No note detected — hold note closer and ensure good lighting.")
        return
    best      = max(dets, key=lambda x: float(x["confidence"]))
    denom     = best["denomination"]
    conf      = float(best["confidence"])
    auth      = result.get("authenticity", "uncertain").strip().lower()
    auth_conf = float(result.get("authenticity_confidence", 0))
    color     = "#6fcf97" if auth == "real" else "#eb5757" if auth == "fake" else "#D9C4B9"
    slot.markdown(f"""
    <div style="margin-top:12px;padding:26px;border:1px solid rgba(217,196,185,.22);
                border-radius:18px;text-align:center;
                background:radial-gradient(circle at 50% 0%,rgba(114,70,59,.28),transparent 65%),
                rgba(34,21,18,.6)">
        <div style="color:#B49E94;font-size:9px;letter-spacing:2.5px;text-transform:uppercase">
            DETECTED DENOMINATION</div>
        <div style="color:#D9C4B9;font-size:52px;font-weight:760;margin:10px 0 4px;line-height:1">
            &#8377;{denom}</div>
        <div style="color:#B49E94;font-size:13px">Detection confidence: {conf:.2f}%</div>
        <hr style="border:none;border-top:1px solid rgba(217,196,185,.13);margin:20px 0">
        <div style="color:#B49E94;font-size:9px;letter-spacing:2.5px;text-transform:uppercase">
            AUTHENTICITY</div>
        <div style="font-size:22px;font-weight:700;margin:8px 0 4px;color:{color}">
            {auth.title()}</div>
        <div style="color:#B49E94;font-size:13px">
            Authenticity confidence: {auth_conf:.2f}%</div>
    </div>""", unsafe_allow_html=True)

# --------------------------------------------------
# VOICE
# --------------------------------------------------
def speak(msg: str) -> None:
    safe_msg = msg.replace("'", "\\'").replace('"', '\\"')
    ts = int(time.time() * 1000)   # unique → forces re-render → fires every call
    components.html(f"""
<script>
/* ts={ts} */
(function() {{
    if (!window.speechSynthesis) return;
    var utt  = new SpeechSynthesisUtterance("{safe_msg}");
    utt.lang  = 'en-IN';
    utt.pitch = 1.1;
    utt.rate  = 0.95;

    function doSpeak() {{
        var vv = window.speechSynthesis.getVoices();
        var f  = vv.find(function(v) {{
            return /female|zira|hazel|susan|heera|raveena|samantha|karen|moira/i.test(v.name);
        }});
        if (!f) f = vv.find(function(v) {{
            return v.lang.startsWith('en') &&
                   !/\\b(male|david|mark|george|james|richard|daniel)\\b/i.test(v.name);
        }});
        if (f) utt.voice = f;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utt);
    }}

    if (window.speechSynthesis.getVoices().length === 0) {{
        window.speechSynthesis.onvoiceschanged = function() {{
            window.speechSynthesis.onvoiceschanged = null;
            doSpeak();
        }};
    }} else {{
        doSpeak();
    }}
}})();
</script>
""", height=1)

# --------------------------------------------------
# MAIN UI — two columns
# --------------------------------------------------
left, right = st.columns([1.2, 0.85], gap="large")

with left:
    st.markdown("#### 📷 Live webcam scanner")
    st.caption("Hold the note **close to the camera so it fills the frame**, then tap **Take Photo**.")

    # facing_mode="environment" → back camera on phones, webcam on laptop
    img = st.camera_input(
        label="Point at a note and take a photo",
        label_visibility="collapsed",
        key="cam_main",
    )

    if img:
        st.success("✅ Photo captured — running detection…")
    else:
        st.info("📷 Allow camera access if prompted, then tap **Take Photo**.")

with right:
    st.markdown("#### 🔍 Latest scan")
    st.caption("Result appears here after each photo.")
    slot = st.empty()

    # Always show last good result while camera is live
    if st.session_state.get("last_result"):
        show_result(st.session_state["last_result"], slot)

    if img is not None:
        with st.spinner("Running AI detection…"):
            result = run_inference(img.getvalue())

        status = result.get("status", "no_note")
        dets   = result.get("denomination_detections", [])

        if status == "detected" and dets:
            # ✅ Genuine note — save and display
            st.session_state["last_result"] = result
            show_result(result, slot)

            best = max(dets, key=lambda x: float(x["confidence"]))
            auth = result.get("authenticity", "uncertain").lower()
            dn   = best["denomination"]

            if auth == "real":
                msg = f"Scan complete. The {dn} rupee note is real."
            elif auth == "fake":
                msg = f"Warning! The {dn} rupee note appears to be fake."
            else:
                msg = f"A {dn} rupee note detected. Authenticity uncertain."
            speak(msg)

        elif status == "unclear":
            # ⚠️ Note-like object seen but too far / low confidence
            slot.warning(
                "⚠️ No currency detected. Make sure the lighting is sufficient "
                "and hold the currency note closer to the camera."
            )

        else:
            # ❌ Nothing that looks like a note at all
            slot.error("❌ No currency note detected.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="site-footer">NOTEVERITY &nbsp;&middot;&nbsp; WEBCAM DETECTION'
    ' &nbsp;&middot;&nbsp; CURRENCY VERIFICATION</div>',
    unsafe_allow_html=True,
)
