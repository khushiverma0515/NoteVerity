import os
import sys

import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

BACKEND_URL           = "http://127.0.0.1:8001/predict"
SCAN_INTERVAL_SECONDS = 5

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="NoteVerity",
    page_icon="💵",
    layout="wide",
)

# --------------------------------------------------
# PAGE-LEVEL CSS (Streamlit shell)
# --------------------------------------------------
st.html("""
<style>
:root {
    --chocolate: #221512;
    --silver:    #D9C4B9;
    --tuscany:   #B49E94;
}
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
.hero-title { color: var(--silver);  font-size: clamp(38px, 6vw, 58px); letter-spacing: -1.8px; font-weight: 760; margin: 14px 0; }
.hero-copy  { color: var(--tuscany); font-size: 15px; opacity: .85; }
.site-footer {
    margin-top: 30px; padding: 20px;
    border-top: 1px solid rgba(217,196,185,.1);
    text-align: center; color: var(--tuscany); opacity: .6; font-size: 11px;
}
</style>

<div class="hero">
    <div class="eyebrow">NOTEVERITY &middot; CURRENCY VERIFICATION</div>
    <div class="hero-title">NoteVerity</div>
    <div class="hero-copy">
        AI-powered currency denomination detection and<br>
        authenticity screening through your webcam.
    </div>
</div>
""")

# --------------------------------------------------
# SELF-CONTAINED CAMERA + DETECTION COMPONENT
# --------------------------------------------------
SCANNER_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: transparent;
    color: #D9C4B9;
}}

.layout {{
    display: flex;
    gap: 28px;
    align-items: flex-start;
}}

.cam-side {{ flex: 1.25; min-width: 0; }}

.cam-wrap {{
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    background: #0d0806;
    aspect-ratio: 4/3;
}}

video {{
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
}}

.scan-pulse {{
    position: absolute; inset: 0;
    border: 2px solid transparent;
    border-radius: 14px;
    pointer-events: none;
    transition: border-color .3s;
}}
.scan-pulse.active {{ border-color: rgba(111,207,151,.6); }}

.countdown {{
    position: absolute; bottom: 10px; right: 12px;
    background: rgba(0,0,0,.55); backdrop-filter: blur(4px);
    color: #D9C4B9; font-size: 11px; padding: 3px 9px;
    border-radius: 20px; pointer-events: none;
}}

.status-bar {{
    margin-top: 10px; padding: 9px 14px;
    border-radius: 10px; font-size: 13px; text-align: center;
}}
.status-bar.ok    {{ background: rgba(111,207,151,.12); color: #6fcf97; border:1px solid rgba(111,207,151,.25); }}
.status-bar.wait  {{ background: rgba(180,158,148,.08); color: #B49E94; border:1px solid rgba(180,158,148,.15); }}
.status-bar.error {{ background: rgba(235,87,87,.12);   color: #eb5757; border:1px solid rgba(235,87,87,.25); }}

.result-side {{ flex: 0.85; min-width: 0; }}

.panel-title {{
    color: #D9C4B9; font-size: 17px; font-weight: 650;
    margin-bottom: 6px;
}}
.panel-sub {{
    color: #B49E94; font-size: 12px; margin-bottom: 14px; line-height: 1.5;
}}

.result-card {{
    padding: 28px 22px;
    border: 1px solid rgba(217,196,185,.2);
    border-radius: 18px;
    background:
        radial-gradient(circle at 50% 0%, rgba(114,70,59,.28), transparent 65%),
        rgba(34,21,18,.6);
    text-align: center;
}}

.r-eyebrow {{
    color: #B49E94; font-size: 9px;
    letter-spacing: 2.5px; opacity: .85;
    text-transform: uppercase;
}}
.r-value {{
    color: #D9C4B9; font-size: 52px;
    font-weight: 760; margin: 10px 0 4px;
    line-height: 1;
}}
.r-conf  {{ color: #B49E94; font-size: 13px; }}
.r-divider {{
    border: none; border-top: 1px solid rgba(217,196,185,.13);
    margin: 20px 0;
}}
.r-auth {{
    font-size: 22px; font-weight: 700; margin: 8px 0 4px;
}}

.idle-card {{
    padding: 50px 20px;
    border: 1px dashed rgba(217,196,185,.15);
    border-radius: 18px;
    text-align: center; color: #B49E94;
}}
.idle-icon {{ font-size: 36px; margin-bottom: 12px; }}
.idle-text {{ font-size: 13px; line-height: 1.6; opacity: .8; }}
</style>
</head>
<body>

<div class="layout">

    <!-- Camera -->
    <div class="cam-side">
        <div class="cam-wrap">
            <video id="vid" autoplay playsinline muted></video>
            <div class="scan-pulse" id="pulse"></div>
            <div class="countdown" id="countdown">-</div>
        </div>
        <div class="status-bar wait" id="status">Starting camera...</div>
    </div>

    <!-- Result -->
    <div class="result-side">
        <div class="panel-title">Latest scan</div>
        <div class="panel-sub">
            Results update every {SCAN_INTERVAL_SECONDS} seconds automatically.<br>
            No interaction needed - just hold the note in view.
        </div>
        <div id="result-area">
            <div class="idle-card">
                <div class="idle-icon">🔍</div>
                <div class="idle-text">
                    Waiting for first scan...<br>
                    Camera is opening automatically.
                </div>
            </div>
        </div>
    </div>

</div>

<canvas id="canvas" style="display:none"></canvas>

<script>
var API      = '{BACKEND_URL}';
var INTERVAL = {SCAN_INTERVAL_SECONDS * 1000};

var vid        = document.getElementById('vid');
var canvas     = document.getElementById('canvas');
var ctx        = canvas.getContext('2d');
var statusBar  = document.getElementById('status');
var resultArea = document.getElementById('result-area');
var pulse      = document.getElementById('pulse');
var cntdEl     = document.getElementById('countdown');

var lastSpoken   = '';
var scanning     = false;
var nextScanTime = 0;
var timerID      = null;

// 1. Open camera instantly
navigator.mediaDevices.getUserMedia({{
    video: {{ width: {{ideal: 640}}, height: {{ideal: 480}}, facingMode: 'environment' }},
    audio: false
}})
.then(function(stream) {{
    vid.srcObject = stream;
    setStatus('ok', 'Camera active - scanning every {SCAN_INTERVAL_SECONDS} s automatically');
    setTimeout(function() {{ scheduleNext(0); }}, 1500);
}})
.catch(function(err) {{
    setStatus('error', 'Camera error: ' + err.message + '. Please allow camera access and refresh.');
}});

// 2. Scheduling + countdown
function scheduleNext(delayMs) {{
    if (timerID) clearTimeout(timerID);
    nextScanTime = Date.now() + delayMs;
    timerID = setTimeout(captureAndSend, delayMs);
    updateCountdown();
}}

function updateCountdown() {{
    var remaining = Math.max(0, Math.ceil((nextScanTime - Date.now()) / 1000));
    cntdEl.textContent = remaining > 0 ? 'Next scan: ' + remaining + ' s' : 'Scanning...';
    if (remaining > 0) setTimeout(updateCountdown, 500);
}}

// 3. Capture frame and POST to backend
function captureAndSend() {{
    if (scanning) return;
    scanning = true;
    pulse.classList.add('active');
    cntdEl.textContent = 'Scanning...';

    canvas.width  = vid.videoWidth  || 640;
    canvas.height = vid.videoHeight || 480;
    ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(function(blob) {{
        var form = new FormData();
        form.append('file', blob, 'frame.jpg');

        fetch(API, {{ method: 'POST', body: form }})
        .then(function(r) {{
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        }})
        .then(function(data) {{
            scanning = false;
            pulse.classList.remove('active');
            setStatus('ok', 'Camera active - scanning every {SCAN_INTERVAL_SECONDS} s automatically');
            showResult(data);
            scheduleNext(INTERVAL);
        }})
        .catch(function(err) {{
            scanning = false;
            pulse.classList.remove('active');
            setStatus('error', 'Backend unreachable: ' + err.message +
                ' - run: uvicorn backend.main:app --port 8001 --reload');
            scheduleNext(INTERVAL);
        }});
    }}, 'image/jpeg', 0.85);
}}

// 4. Render result card
function showResult(data) {{
    var dets = data.denomination_detections || [];

    if (!dets.length) {{
        setStatus('wait', 'No note detected - hold note closer and steady');
        return;
    }}

    var best     = dets.reduce(function(a, b) {{ return +a.confidence > +b.confidence ? a : b; }});
    var denom    = best.denomination;
    var conf     = (+best.confidence).toFixed(2);
    var auth     = (data.authenticity || 'uncertain').toLowerCase();
    var authConf = (+(data.authenticity_confidence || 0)).toFixed(2);
    var color    = auth === 'real' ? '#6fcf97' : auth === 'fake' ? '#eb5757' : '#D9C4B9';
    var authCap  = auth.charAt(0).toUpperCase() + auth.slice(1);

    resultArea.innerHTML =
        '<div class="result-card">'                                                +
        '  <div class="r-eyebrow">DETECTED DENOMINATION</div>'                    +
        '  <div class="r-value">&#8377;' + denom + '</div>'                       +
        '  <div class="r-conf">Detection confidence: ' + conf + '%</div>'         +
        '  <hr class="r-divider">'                                                +
        '  <div class="r-eyebrow">AUTHENTICITY</div>'                             +
        '  <div class="r-auth" style="color:' + color + '">' + authCap + '</div>' +
        '  <div class="r-conf">Authenticity confidence: ' + authConf + '%</div>'  +
        '</div>';

    // 5. Female voice via Web Speech API
    var sig = denom + '|' + auth;
    if (sig !== lastSpoken && window.speechSynthesis) {{
        lastSpoken = sig;
        window.speechSynthesis.cancel();

        var utt   = new SpeechSynthesisUtterance();
        utt.lang  = 'en-IN';
        utt.pitch = 1.1;
        utt.rate  = 0.95;

        function speakWithFemaleVoice() {{
            var voices = window.speechSynthesis.getVoices();

            // 1st priority: explicitly female or known female voice names
            var female = voices.find(function(v) {{
                return /female|woman|zira|hazel|susan|heera|raveena|victoria|samantha|karen|moira/i.test(v.name);
            }});

            // 2nd priority: any English voice that is not male-named
            if (!female) {{
                female = voices.find(function(v) {{
                    return v.lang.startsWith('en') &&
                           !/male|david|mark|george|james|richard|daniel/i.test(v.name);
                }});
            }}

            if (female) utt.voice = female;

            if (auth === 'real')
                utt.text = 'Scan complete. The detected' + denom + ' rupee note is real.';
            else if (auth === 'fake')
                utt.text = 'Warning! The detected ' + denom + ' rupee note appears to be fake.';
            else
                utt.text = 'A ' + denom + ' rupee note detected. Authenticity uncertain.';

            window.speechSynthesis.speak(utt);
        }}

        // Some browsers load voices async — wait if needed
        if (window.speechSynthesis.getVoices().length === 0) {{
            window.speechSynthesis.onvoiceschanged = function() {{
                window.speechSynthesis.onvoiceschanged = null;
                speakWithFemaleVoice();
            }};
        }} else {{
            speakWithFemaleVoice();
        }}
    }}
}}

// Helper
function setStatus(type, msg) {{
    statusBar.className   = 'status-bar ' + type;
    statusBar.textContent = msg;
}}
</script>
</body>
</html>
"""

components.html(SCANNER_HTML, height=520, scrolling=False)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="site-footer">NOTEVERITY &nbsp;&middot;&nbsp; WEBCAM DETECTION &nbsp;&middot;&nbsp; CURRENCY VERIFICATION</div>',
    unsafe_allow_html=True,
)