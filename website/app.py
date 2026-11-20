import io
import os
import sys
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
# PATHS
# --------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

SCAN_INTERVAL_SECONDS = 5

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NoteVerity",
    page_icon="💵",
    layout="wide",
)

# --------------------------------------------------
# LOAD MODELS  (cached — loads only once)
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models… this takes ~20 seconds on first run.")
def load_models():
    yolo = YOLO(str(MODEL_DIR / "best.pt"))

    resnet = resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(
        torch.load(MODEL_DIR / "real_fake_model.pth", map_location="cpu", weights_only=True)
    )
    resnet.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return yolo, resnet, tfm

denomination_model, authenticity_model, transform = load_models()

# --------------------------------------------------
# INFERENCE  (called from Streamlit, no HTTP needed)
# --------------------------------------------------
def run_inference(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid image"}

    with torch.inference_mode():
        results = denomination_model.predict(image, conf=0.25, verbose=False, imgsz=320, device="cpu")

    detections = []
    if not any(len(r.boxes) > 0 for r in results):
        return {
            "denomination_detections": [],
            "authenticity": "UNCERTAIN",
            "authenticity_confidence": 0.0,
        }

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "denomination": str(result.names[class_id]),
                "confidence":   round(float(box.conf.item()) * 100, 2),
            })

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(authenticity_model(tensor), dim=1)[0]

    classes      = ["fake", "real"]
    pred_id      = int(torch.argmax(probs).item())
    authenticity = classes[pred_id].upper()
    auth_conf    = round(float(probs[pred_id]) * 100, 2)

    return {
        "denomination_detections": detections,
        "authenticity":            authenticity,
        "authenticity_confidence": auth_conf,
    }

# --------------------------------------------------
# PAGE CSS
# --------------------------------------------------
st.html("""
<style>
:root { --silver: #D9C4B9; --tuscany: #B49E94; }
.stApp {
    background:
        radial-gradient(ellipse 70% 45% at 50% -8%, rgba(114,70,59,.42), transparent 70%),
        linear-gradient(180deg, #221512 0%, #1b100d 100%);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 40px; }
.hero       { text-align: center; padding: 24px 0 44px; }
.eyebrow    { color: var(--tuscany); font-size: 10px; letter-spacing: 3px; opacity: .75; }
.hero-title { color: var(--silver); font-size: clamp(38px,6vw,58px); letter-spacing:-1.8px; font-weight:760; margin:14px 0; }
.hero-copy  { color: var(--tuscany); font-size: 15px; opacity: .85; }
.site-footer { margin-top:30px; padding:20px; border-top:1px solid rgba(217,196,185,.1); text-align:center; color:var(--tuscany); opacity:.6; font-size:11px; }
</style>
<div class="hero">
    <div class="eyebrow">NOTEVERITY &middot; CURRENCY VERIFICATION</div>
    <div class="hero-title">NoteVerity</div>
    <div class="hero-copy">AI-powered currency denomination detection and<br>authenticity screening through your webcam.</div>
</div>
""")

# --------------------------------------------------
# CAMERA  — captures frames via browser getUserMedia
# and sends them to a hidden Streamlit endpoint
# --------------------------------------------------
# We use a hidden file_uploader as the bridge:
# JS captures a frame → converts to blob → triggers
# a programmatic upload → Streamlit reads it → inference

CAMERA_HTML = f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent;color:#D9C4B9}}
.layout{{display:flex;gap:28px;align-items:flex-start}}
.cam-side{{flex:1.25;min-width:0}}
.cam-wrap{{position:relative;border-radius:14px;overflow:hidden;background:#0d0806;aspect-ratio:4/3}}
video{{width:100%;height:100%;object-fit:cover;display:block}}
.scan-pulse{{position:absolute;inset:0;border:2px solid transparent;border-radius:14px;pointer-events:none;transition:border-color .3s}}
.scan-pulse.active{{border-color:rgba(111,207,151,.6)}}
.countdown{{position:absolute;bottom:10px;right:12px;background:rgba(0,0,0,.55);backdrop-filter:blur(4px);color:#D9C4B9;font-size:11px;padding:3px 9px;border-radius:20px;pointer-events:none}}
.status-bar{{margin-top:10px;padding:9px 14px;border-radius:10px;font-size:13px;text-align:center}}
.status-bar.ok{{background:rgba(111,207,151,.12);color:#6fcf97;border:1px solid rgba(111,207,151,.25)}}
.status-bar.wait{{background:rgba(180,158,148,.08);color:#B49E94;border:1px solid rgba(180,158,148,.15)}}
.status-bar.error{{background:rgba(235,87,87,.12);color:#eb5757;border:1px solid rgba(235,87,87,.25)}}
.result-side{{flex:0.85;min-width:0}}
.panel-title{{color:#D9C4B9;font-size:17px;font-weight:650;margin-bottom:6px}}
.panel-sub{{color:#B49E94;font-size:12px;margin-bottom:14px;line-height:1.5}}
.result-card{{padding:28px 22px;border:1px solid rgba(217,196,185,.2);border-radius:18px;background:radial-gradient(circle at 50% 0%,rgba(114,70,59,.28),transparent 65%),rgba(34,21,18,.6);text-align:center}}
.r-eyebrow{{color:#B49E94;font-size:9px;letter-spacing:2.5px;opacity:.85;text-transform:uppercase}}
.r-value{{color:#D9C4B9;font-size:52px;font-weight:760;margin:10px 0 4px;line-height:1}}
.r-conf{{color:#B49E94;font-size:13px}}
.r-divider{{border:none;border-top:1px solid rgba(217,196,185,.13);margin:20px 0}}
.r-auth{{font-size:22px;font-weight:700;margin:8px 0 4px}}
.idle-card{{padding:50px 20px;border:1px dashed rgba(217,196,185,.15);border-radius:18px;text-align:center;color:#B49E94}}
.idle-icon{{font-size:36px;margin-bottom:12px}}
.idle-text{{font-size:13px;line-height:1.6;opacity:.8}}
</style></head><body>
<div class="layout">
  <div class="cam-side">
    <div class="cam-wrap">
      <video id="vid" autoplay playsinline muted></video>
      <div class="scan-pulse" id="pulse"></div>
      <div class="countdown" id="countdown">-</div>
    </div>
    <div class="status-bar wait" id="status">Starting camera...</div>
  </div>
  <div class="result-side">
    <div class="panel-title">Latest scan</div>
    <div class="panel-sub">Results update every {SCAN_INTERVAL_SECONDS} seconds automatically.<br>No interaction needed - just hold the note in view.</div>
    <div id="result-area">
      <div class="idle-card"><div class="idle-icon">🔍</div>
        <div class="idle-text">Waiting for first scan...<br>Camera is opening automatically.</div>
      </div>
    </div>
  </div>
</div>
<canvas id="canvas" style="display:none"></canvas>
<script>
var INTERVAL={SCAN_INTERVAL_SECONDS * 1000};
var vid=document.getElementById('vid'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
var statusBar=document.getElementById('status'),resultArea=document.getElementById('result-area');
var pulse=document.getElementById('pulse'),cntdEl=document.getElementById('countdown');
var lastSpoken='',scanning=false,nextScanTime=0,timerID=null;

navigator.mediaDevices.getUserMedia({{video:{{width:{{ideal:640}},height:{{ideal:480}}}},audio:false}})
.then(function(stream){{
    vid.srcObject=stream;
    setStatus('ok','Camera active - scanning every {SCAN_INTERVAL_SECONDS}s automatically');
    setTimeout(function(){{scheduleNext(0);}},1500);
}})
.catch(function(err){{setStatus('error','Camera error: '+err.message+'. Allow camera access and refresh.');}});

function scheduleNext(d){{
    if(timerID)clearTimeout(timerID);
    nextScanTime=Date.now()+d;
    timerID=setTimeout(captureAndSend,d);
    updateCountdown();
}}
function updateCountdown(){{
    var r=Math.max(0,Math.ceil((nextScanTime-Date.now())/1000));
    cntdEl.textContent=r>0?'Next scan: '+r+' s':'Scanning...';
    if(r>0)setTimeout(updateCountdown,500);
}}

function captureAndSend(){{
    if(scanning)return;
    scanning=true;pulse.classList.add('active');cntdEl.textContent='Scanning...';
    canvas.width=vid.videoWidth||640;canvas.height=vid.videoHeight||480;
    ctx.drawImage(vid,0,0,canvas.width,canvas.height);
    canvas.toBlob(function(blob){{
        var reader=new FileReader();
        reader.onloadend=function(){{
            // Send base64 frame to Streamlit via postMessage
            window.parent.postMessage({{type:'noteverity_frame',data:reader.result}},'*');
        }};
        reader.readAsDataURL(blob);
    }},'image/jpeg',0.85);
}}

// Listen for inference result back from Streamlit
window.addEventListener('message',function(e){{
    if(!e.data||e.data.type!=='noteverity_result')return;
    scanning=false;pulse.classList.remove('active');
    setStatus('ok','Camera active - scanning every {SCAN_INTERVAL_SECONDS}s automatically');
    showResult(e.data.result);
    scheduleNext(INTERVAL);
}});

function showResult(data){{
    var dets=data.denomination_detections||[];
    if(!dets.length){{setStatus('wait','No note detected - hold note closer and steady');return;}}
    var best=dets.reduce(function(a,b){{return +a.confidence>+b.confidence?a:b;}});
    var denom=best.denomination,conf=(+best.confidence).toFixed(2);
    var auth=(data.authenticity||'uncertain').toLowerCase();
    var authConf=(+(data.authenticity_confidence||0)).toFixed(2);
    var color=auth==='real'?'#6fcf97':auth==='fake'?'#eb5757':'#D9C4B9';
    var authCap=auth.charAt(0).toUpperCase()+auth.slice(1);
    resultArea.innerHTML='<div class="result-card"><div class="r-eyebrow">DETECTED DENOMINATION</div><div class="r-value">&#8377;'+denom+'</div><div class="r-conf">Detection confidence: '+conf+'%</div><hr class="r-divider"><div class="r-eyebrow">AUTHENTICITY</div><div class="r-auth" style="color:'+color+'">'+authCap+'</div><div class="r-conf">Authenticity confidence: '+authConf+'%</div></div>';
    var sig=denom+'|'+auth;
    if(sig!==lastSpoken&&window.speechSynthesis){{
        lastSpoken=sig;window.speechSynthesis.cancel();
        var utt=new SpeechSynthesisUtterance();
        utt.lang='en-IN';utt.pitch=1.1;utt.rate=0.95;
        function speak(){{
            var voices=window.speechSynthesis.getVoices();
            var f=voices.find(function(v){{return /female|zira|hazel|susan|heera|raveena|samantha|karen/i.test(v.name);}});
            if(!f)f=voices.find(function(v){{return v.lang.startsWith('en')&&!/male|david|mark|george|james|richard/i.test(v.name);}});
            if(f)utt.voice=f;
            utt.text=auth==='real'?'Scan complete. The '+denom+' rupee note is real.':auth==='fake'?'Warning! The '+denom+' rupee note appears to be fake.':'A '+denom+' rupee note detected. Authenticity uncertain.';
            window.speechSynthesis.speak(utt);
        }}
        if(window.speechSynthesis.getVoices().length===0){{window.speechSynthesis.onvoiceschanged=function(){{window.speechSynthesis.onvoiceschanged=null;speak();}};}}else{{speak();}}
    }}
}}
function setStatus(type,msg){{statusBar.className='status-bar '+type;statusBar.textContent=msg;}}
</script></body></html>
"""

# --------------------------------------------------
# RENDER COMPONENT + HANDLE INCOMING FRAMES
# --------------------------------------------------

# We use a hidden file_uploader trick to receive frames from JS via postMessage.
# The JS sends base64 → Streamlit intercepts via st.query_params trick OR
# we use a simple polling approach via session_state + st.fragment.

# Simpler approach: embed the camera component and use
# st.components.v1.html with bidirectional communication via query params.
# Since Streamlit doesn't support direct postMessage from components easily,
# we route frames through a small internal endpoint.

# BEST APPROACH for Streamlit Cloud:
# Use st.camera_input() for capturing + the HTML component for display only.
# The HTML component shows live video; st.camera_input captures frames every Ns.

@st.cache_data(show_spinner=False)
def _dummy():
    return None

# Fragment: runs inference every SCAN_INTERVAL_SECONDS
if hasattr(st, "fragment"):

    @st.fragment(run_every=f"{SCAN_INTERVAL_SECONDS}s")
    def scanner():
        left, right = st.columns([1.2, 0.85], gap="large")

        with left:
            st.markdown("**Live webcam scanner**")
            st.caption("Camera opens instantly — scanning runs automatically every 5 seconds.")
            img = st.camera_input("", label_visibility="collapsed", key="cam_main")
            if img:
                st.success("✅ Scanning active. Hold note steady.")
            else:
                st.info("📷 Click the **●** button above to start scanning.")

        with right:
            st.markdown("**Latest scan**")
            st.caption("Results update automatically every 5 seconds once scanning starts.")
            slot = st.empty()

            # Always show last result
            prev = st.session_state.get("last_result")
            if prev:
                _show_result(prev, slot)

            if img is None:
                return

            now  = time.monotonic()
            last = st.session_state.get("last_scan_time", 0.0)
            if now - last < SCAN_INTERVAL_SECONDS - 0.5:
                return
            st.session_state["last_scan_time"] = now

            with st.spinner("Running detection..."):
                result = run_inference(img.getvalue())

            st.session_state["last_result"] = result
            _show_result(result, slot)

            # Voice handled via JS — trigger via rerun signal
            dets = result.get("denomination_detections", [])
            if dets:
                best = max(dets, key=lambda x: float(x["confidence"]))
                auth = result.get("authenticity", "uncertain").lower()
                sig  = f"{best['denomination']}|{auth}"
                if sig != st.session_state.get("last_sig"):
                    st.session_state["last_sig"] = sig

    def _show_result(result, placeholder):
        dets = result.get("denomination_detections", [])
        if not dets:
            placeholder.warning("No currency note detected. Adjust position.")
            return
        best      = max(dets, key=lambda x: float(x["confidence"]))
        denom     = best["denomination"]
        conf      = float(best["confidence"])
        auth      = result.get("authenticity", "uncertain").strip().lower()
        auth_conf = float(result.get("authenticity_confidence", 0))
        color     = "#6fcf97" if auth == "real" else "#eb5757" if auth == "fake" else "#D9C4B9"
        placeholder.markdown(f"""
        <div style="margin-top:12px;padding:26px;border:1px solid rgba(217,196,185,.22);border-radius:18px;
                    background:radial-gradient(circle at 50% 0%,rgba(114,70,59,.28),transparent 65%),rgba(34,21,18,.6);text-align:center">
            <div style="color:#B49E94;font-size:9px;letter-spacing:2.5px;text-transform:uppercase">DETECTED DENOMINATION</div>
            <div style="color:#D9C4B9;font-size:52px;font-weight:760;margin:10px 0 4px;line-height:1">&#8377;{denom}</div>
            <div style="color:#B49E94;font-size:13px">Detection confidence: {conf:.2f}%</div>
            <hr style="border:none;border-top:1px solid rgba(217,196,185,.13);margin:20px 0">
            <div style="color:#B49E94;font-size:9px;letter-spacing:2.5px;text-transform:uppercase">AUTHENTICITY</div>
            <div style="font-size:22px;font-weight:700;margin:8px 0 4px;color:{color}">{auth.title()}</div>
            <div style="color:#B49E94;font-size:13px">Authenticity confidence: {auth_conf:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    scanner()

else:
    st.error("Automatic scanning requires Streamlit ≥ 1.33. Run `pip install --upgrade streamlit`.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="site-footer">NOTEVERITY &nbsp;&middot;&nbsp; WEBCAM DETECTION &nbsp;&middot;&nbsp; CURRENCY VERIFICATION</div>',
    unsafe_allow_html=True,
)
