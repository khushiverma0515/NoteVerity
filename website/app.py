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
SCAN_INTERVAL_SECONDS = 7       # seconds between auto-scans
MIN_DENOM_CONF        = 60.0    # ignore detections below this % — kills false positives

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

    # conf=0.60 — only accept high-confidence YOLO detections to kill false positives
    with torch.inference_mode():
        results = denomination_model.predict(
            image, conf=0.60, verbose=False, imgsz=416, device="cpu"
        )

    # Collect detections above threshold
    detections = []
    for result in results:
        for box in result.boxes:
            conf_pct = round(float(box.conf.item()) * 100, 2)
            if conf_pct >= MIN_DENOM_CONF:          # extra safety filter
                class_id = int(box.cls.item())
                detections.append({
                    "denomination": str(result.names[class_id]),
                    "confidence":   conf_pct,
                })

    if not detections:
        return {
            "denomination_detections": [],
            "authenticity":            "UNCERTAIN",
            "authenticity_confidence": 0.0,
        }

    # Only run real/fake if we have a confident note detection
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
# AUTO-CLICKER — clicks Take Photo automatically every 7 s.
# After clicking, button dims for 6 s so user knows scan
# is in progress, then restores.
# --------------------------------------------------
components.html("""
<script>
(function() {
    var SHOOT_INTERVAL = 7000;
    var HIDE_DURATION  = 6000;

    function setup() {
        var p;
        try { p = window.parent.document; } catch(e) { return; }

        var styleId = '_nv_btn_hide_';
        if (!p.getElementById(styleId)) {
            var s = p.createElement('style');
            s.id = styleId;
            p.head && p.head.appendChild(s);
        }

        function hideBtn() {
            var s = p.getElementById(styleId);
            if (!s) return;
            s.textContent =
                '[data-testid="stCameraInputButton"]{' +
                    'opacity:0.35!important;pointer-events:none!important;' +
                '}';
            setTimeout(function() { s.textContent = ''; }, HIDE_DURATION);
        }

        function shoot() {
            var clearBtn = p.querySelector('[data-testid="stCameraInputClear"]');
            if (clearBtn) { clearBtn.click(); return; }
            var shutterBtn = p.querySelector('[data-testid="stCameraInputButton"]');
            if (shutterBtn) { shutterBtn.click(); hideBtn(); }
        }

        setTimeout(shoot, 2500);
        setInterval(shoot, SHOOT_INTERVAL);
    }

    setup();
    setTimeout(setup, 800);
})();
</script>
""", height=0)

# --------------------------------------------------
# RESULT RENDERER
# --------------------------------------------------
def show_result(result: dict, slot) -> None:
    dets = result.get("denomination_detections", [])
    if not dets:
        slot.warning("No note detected — hold note closer and ensure good lighting.")
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
    ts = int(time.time() * 1000)
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
# SCANNER FRAGMENT  — reruns every 7 s automatically
# --------------------------------------------------
@st.fragment(run_every=f"{SCAN_INTERVAL_SECONDS}s")
def scanner():
    left, right = st.columns([1.2, 0.85], gap="large")

    with left:
        st.markdown("#### Live webcam scanner")
        st.caption(
            "Camera opens automatically — scanning every 7 seconds. No interaction needed."
        )
        img = st.camera_input("", label_visibility="collapsed", key="cam_main")
        if img:
            st.success("✅ Frame captured. Sending to AI...")
        else:
            st.info("📷 Allow camera access if prompted — scanning starts automatically.")

    with right:
        st.markdown("#### Latest scan")
        st.caption("Results update every 7 seconds.")
        slot = st.empty()

        # Always show last good result while waiting for next scan
        if st.session_state.get("last_result"):
            show_result(st.session_state["last_result"], slot)

        if img is None:
            return

        # Throttle — don't run inference more than once per interval
        now  = time.monotonic()
        last = st.session_state.get("last_scan_time", 0.0)
        if now - last < SCAN_INTERVAL_SECONDS - 0.5:
            return
        st.session_state["last_scan_time"] = now

        with st.spinner("Running detection..."):
            result = run_inference(img.getvalue())

        dets = result.get("denomination_detections", [])

        if dets:
            # Only update stored result when we have a REAL confident detection
            st.session_state["last_result"] = result
            show_result(result, slot)

            best = max(dets, key=lambda x: float(x["confidence"]))
            auth = result.get("authenticity", "uncertain").lower()
            dn   = best["denomination"]

            # Voice fires every confident detection (ts makes it unique every time)
            if auth == "real":
                msg = f"Scan complete. The {dn} rupee note is real."
            elif auth == "fake":
                msg = f"Warning! The {dn} rupee note appears to be fake."
            else:
                msg = f"A {dn} rupee note detected. Authenticity uncertain."
            speak(msg)
        else:
            # No confident note found — show warning in result slot
            # but do NOT overwrite last_result so the last good scan stays in memory
            slot.warning("No note detected — hold note closer and ensure good lighting.")


scanner()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="site-footer">NOTEVERITY &nbsp;&middot;&nbsp; WEBCAM DETECTION'
    ' &nbsp;&middot;&nbsp; CURRENCY VERIFICATION</div>',
    unsafe_allow_html=True,
)
