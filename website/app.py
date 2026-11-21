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
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
SCAN_INTERVAL_SECONDS = 5

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
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid image"}

    with torch.inference_mode():
        # imgsz=320 for better detection accuracy
        results = denomination_model.predict(
            image, conf=0.25, verbose=False, imgsz=320, device="cpu"
        )

    if not any(len(r.boxes) > 0 for r in results):
        return {
            "denomination_detections": [],
            "authenticity": "UNCERTAIN",
            "authenticity_confidence": 0.0,
        }

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "denomination": str(result.names[class_id]),
                "confidence":   round(float(box.conf.item()) * 100, 2),
            })

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs   = torch.softmax(authenticity_model(tensor), dim=1)[0]
    classes = ["fake", "real"]
    pred_id = int(torch.argmax(probs).item())

    return {
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
.site-footer { margin-top:30px; padding:20px; border-top:1px solid rgba(217,196,185,.1);
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
        slot.warning("No note detected — hold note closer, ensure good lighting.")
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
# VOICE via components.html  (NOT st.html — that's sandboxed)
# --------------------------------------------------
def speak(msg: str) -> None:
    """Trigger female voice output using components.html which allows audio."""
    safe_msg = msg.replace("'", "\\'").replace('"', '\\"')
    components.html(f"""
<script>
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
""", height=0)

# --------------------------------------------------
# SCANNER FRAGMENT  — reruns every 5 s automatically
# --------------------------------------------------
@st.fragment(run_every=f"{SCAN_INTERVAL_SECONDS}s")
def scanner():
    left, right = st.columns([1.2, 0.85], gap="large")

    with left:
        st.markdown("#### Live webcam scanner")
        st.caption("Hold the note clearly in view — scanning every 5 seconds automatically.")
        img = st.camera_input("", label_visibility="collapsed", key="cam_main")
        if img:
            st.success("✅ Frame captured. Sending to AI...")
        else:
            st.info("📷 Allow camera access if prompted.")

    with right:
        st.markdown("#### Latest scan")
        st.caption("Results update every 5 seconds.")
        slot = st.empty()

        # Always show last result while waiting for new one
        if st.session_state.get("last_result"):
            show_result(st.session_state["last_result"], slot)

        if img is None:
            return

        # Rate-limit to avoid hammering the model on every rerun
        now  = time.monotonic()
        last = st.session_state.get("last_scan_time", 0.0)
        if now - last < SCAN_INTERVAL_SECONDS - 0.5:
            return
        st.session_state["last_scan_time"] = now

        with st.spinner("Running detection..."):
            result = run_inference(img.getvalue())

        st.session_state["last_result"] = result
        show_result(result, slot)

        # Voice — only speaks when denomination/authenticity changes
        dets = result.get("denomination_detections", [])
        if dets:
            best = max(dets, key=lambda x: float(x["confidence"]))
            auth = result.get("authenticity", "uncertain").lower()
            sig  = f"{best['denomination']}|{auth}"

            if sig != st.session_state.get("last_sig"):
                st.session_state["last_sig"] = sig
                dn = best["denomination"]
                if auth == "real":
                    msg = f"Scan complete. The {dn} rupee note is real."
                elif auth == "fake":
                    msg = f"Warning! The {dn} rupee note appears to be fake."
                else:
                    msg = f"A {dn} rupee note detected. Authenticity uncertain."
                speak(msg)


scanner()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="site-footer">NOTEVERITY &nbsp;&middot;&nbsp; WEBCAM DETECTION'
    ' &nbsp;&middot;&nbsp; CURRENCY VERIFICATION</div>',
    unsafe_allow_html=True,
)
