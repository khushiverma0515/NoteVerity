import streamlit as st
import cv2
import numpy as np
import sys
import os
from gtts import gTTS
import base64
import streamlit.components.v1 as components

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import detect_currency
from model.predict_real_fake import predict_real_fake

st.set_page_config(
    page_title="Indian Currency Note Detection",
    page_icon="💵",
    layout="wide"
)

st.html("""
<style>

/* =========================================================
   INDIAN CURRENCY AI — PREMIUM BROWN DESIGN SYSTEM
   ========================================================= */

/* ---------- COLOUR PALETTE ---------- */

:root {
    --chocolate: #221512;
    --bole: #72463B;
    --silver: #D9C4B9;
    --tuscany: #B49E94;
    --metallic: #8B8282;

    --glass: rgba(114, 70, 59, 0.18);
    --glass-strong: rgba(114, 70, 59, 0.30);
    --border: rgba(217, 196, 185, 0.18);
    --border-strong: rgba(217, 196, 185, 0.34);
}


/* =========================================================
   PAGE
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            ellipse 70% 45% at 50% -8%,
            rgba(114, 70, 59, 0.42),
            transparent 70%
        ),
        radial-gradient(
            ellipse 50% 35% at 0% 45%,
            rgba(180, 158, 148, 0.08),
            transparent 70%
        ),
        linear-gradient(
            180deg,
            #221512 0%,
            #1d120f 48%,
            #160d0b 100%
        );

    color: var(--silver);
}


/* Keep the main page centred */

.block-container {
    max-width: 1180px !important;
    padding-top: 0 !important;
    padding-bottom: 80px !important;
}


/* Remove Streamlit chrome */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* =========================================================
   GLOBAL TEXT
   ========================================================= */

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color: var(--silver);
}


/* =========================================================
   HERO
   ========================================================= */

.hero-section {
    position: relative;

    max-width: 1050px;
    margin: 0 auto;

    padding: 110px 30px 95px;

    text-align: center;
}


/* Large warm glow */

.hero-section::before {
    content: "";

    position: absolute;

    width: 720px;
    height: 420px;

    left: 50%;
    top: 20px;

    transform: translateX(-50%);

    background:
        radial-gradient(
            ellipse,
            rgba(114,70,59,0.48),
            rgba(114,70,59,0.16) 38%,
            transparent 72%
        );

    filter: blur(20px);

    pointer-events: none;
}


/* Small decorative line */

.hero-section::after {
    content: "";

    position: absolute;

    width: 70px;
    height: 1px;

    left: 50%;
    bottom: 45px;

    transform: translateX(-50%);

    background: var(--tuscany);

    opacity: 0.45;
}


.hero-content {
    position: relative;
    z-index: 2;
}


/* Badge */

.hero-badge {
    display: inline-flex;

    align-items: center;
    gap: 10px;

    padding: 9px 17px;

    margin-bottom: 30px;

    border: 1px solid var(--border-strong);
    border-radius: 999px;

    background:
        rgba(114,70,59,0.22);

    color: var(--tuscany);

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 3px;

    box-shadow:
        0 8px 30px rgba(34,21,18,0.35),
        inset 0 1px 0 rgba(217,196,185,0.08);
}


.hero-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: var(--tuscany);

    box-shadow:
        0 0 12px rgba(180,158,148,0.7);
}


/* Main title */

.hero-title {
    font-size: clamp(58px, 8vw, 94px);

    font-weight: 800;

    line-height: 0.94;

    letter-spacing: -5px;

    margin: 0 auto 30px;

    max-width: 900px;

    color: var(--silver);

    text-shadow:
        0 12px 45px rgba(0,0,0,0.35);
}


.hero-title span {
    color: var(--tuscany);

    opacity: 0.85;
}


/* Description */

.hero-description {
    max-width: 620px;

    margin: 0 auto;

    font-size: 16px;

    line-height: 1.75;

    color: var(--tuscany);

    opacity: 0.82;
}


/* Hero meta */

.hero-meta {
    margin-top: 32px;

    display: flex;

    justify-content: center;
    align-items: center;

    gap: 14px;

    color: var(--metallic);

    font-size: 10px;

    font-weight: 650;

    letter-spacing: 2px;
}


.hero-meta-line {
    width: 38px;
    height: 1px;

    background: var(--tuscany);

    opacity: 0.4;
}


/* =========================================================
   CURRENCY DENOMINATIONS
   ========================================================= */

.currency-section {
    max-width: 980px;

    margin: 0 auto 100px;
}


.currency-header {
    text-align: center;

    margin-bottom: 32px;
}


.currency-label {
    color: var(--tuscany);

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 3px;

    margin-bottom: 12px;
}


.currency-title {
    color: var(--silver);

    font-size: 30px;

    font-weight: 750;

    letter-spacing: -1px;

    margin-bottom: 9px;
}


.currency-subtitle {
    color: var(--metallic);

    font-size: 13px;
}


.currency-list {
    display: grid;

    grid-template-columns:
        repeat(6, 1fr);

    gap: 12px;
}


.currency-chip {
    min-height: 82px;

    display: flex;

    align-items: center;
    justify-content: center;

    border: 1px solid var(--border);

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.32),
            rgba(34,21,18,0.58)
        );

    color: var(--silver);

    font-size: 18px;

    font-weight: 700;

    box-shadow:
        0 12px 30px rgba(34,21,18,0.30),
        inset 0 1px 0 rgba(217,196,185,0.06);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        background 0.25s ease,
        box-shadow 0.25s ease;
}


.currency-chip:hover {
    transform: translateY(-5px);

    border-color: var(--border-strong);

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.50),
            rgba(34,21,18,0.62)
        );

    box-shadow:
        0 18px 38px rgba(34,21,18,0.45),
        0 0 25px rgba(114,70,59,0.18);
}


/* =========================================================
   TECHNOLOGY STRIP
   ========================================================= */

.tech-strip {
    max-width: 900px;

    margin: -40px auto 90px;

    display: flex;

    justify-content: center;
    align-items: center;

    gap: 14px;

    flex-wrap: wrap;
}


.tech-item {
    display: inline-flex;

    align-items: center;
    gap: 8px;

    padding: 9px 15px;

    border: 1px solid rgba(180,158,148,0.14);

    border-radius: 999px;

    background:
        rgba(114,70,59,0.14);

    color: var(--tuscany);

    font-size: 9px;

    font-weight: 700;

    letter-spacing: 1.5px;
}


.tech-dot {
    width: 5px;
    height: 5px;

    border-radius: 50%;

    background: var(--tuscany);
}


.tech-separator {
    width: 1px;
    height: 18px;

    background: rgba(180,158,148,0.18);
}


/* =========================================================
   WORKFLOW
   ========================================================= */

.workflow-section {
    padding: 20px 0 100px;
}


.workflow-header {
    text-align: center;

    margin-bottom: 42px;
}


.workflow-eyebrow {
    color: var(--tuscany);

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 3px;

    margin-bottom: 12px;
}


.workflow-title {
    color: var(--silver);

    font-size: 38px;

    font-weight: 750;

    letter-spacing: -1.5px;

    margin-bottom: 10px;
}


.workflow-subtitle {
    color: var(--metallic);

    font-size: 14px;
}


.workflow-grid {
    display: flex;

    gap: 18px;
}


.workflow-card {
    position: relative;

    flex: 1;

    min-height: 220px;

    padding: 30px;

    border: 1px solid var(--border);

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.30),
            rgba(34,21,18,0.60)
        );

    box-shadow:
        0 18px 45px rgba(34,21,18,0.32),
        inset 0 1px 0 rgba(217,196,185,0.05);

    overflow: hidden;

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}


.workflow-card::before {
    content: "";

    position: absolute;

    width: 160px;
    height: 160px;

    right: -70px;
    bottom: -80px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(114,70,59,0.45),
            transparent 70%
        );
}


.workflow-card:hover {
    transform: translateY(-7px);

    border-color: var(--border-strong);

    box-shadow:
        0 25px 55px rgba(34,21,18,0.48),
        0 0 30px rgba(114,70,59,0.12);
}


.workflow-number {
    color: var(--tuscany);

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 2px;

    opacity: 0.65;

    margin-bottom: 40px;
}


.workflow-icon {
    position: absolute;

    top: 25px;
    right: 25px;

    width: 42px;
    height: 42px;

    display: flex;

    align-items: center;
    justify-content: center;

    border: 1px solid var(--border-strong);

    border-radius: 13px;

    background:
        rgba(114,70,59,0.28);

    color: var(--silver);

    font-size: 17px;

    box-shadow:
        inset 0 1px 0 rgba(217,196,185,0.07);
}


.workflow-card-title {
    color: var(--silver);

    font-size: 21px;

    font-weight: 700;

    margin-bottom: 9px;
}


.workflow-card-text {
    max-width: 280px;

    color: var(--tuscany);

    font-size: 13px;

    line-height: 1.65;

    opacity: 0.75;
}


/* =========================================================
   UPLOAD SECTION
   ========================================================= */

.upload-section {
    max-width: 860px;

    margin: 0 auto 25px;
}


.upload-heading {
    text-align: center;

    margin-bottom: 30px;
}


.upload-eyebrow {
    color: var(--tuscany);

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 3px;

    margin-bottom: 12px;
}


.upload-title {
    color: var(--silver);

    font-size: 38px;

    font-weight: 750;

    letter-spacing: -1.5px;

    margin-bottom: 10px;
}


.upload-subtitle {
    color: var(--metallic);

    font-size: 14px;
}


/* Main upload card */

.upload-card {
    position: relative;

    padding: 60px 30px 50px;

    border: 1px dashed rgba(217,196,185,0.36);

    border-radius: 28px;

    background:
        radial-gradient(
            ellipse at 50% 0%,
            rgba(114,70,59,0.50),
            transparent 65%
        ),
        linear-gradient(
            145deg,
            rgba(114,70,59,0.22),
            rgba(34,21,18,0.70)
        );

    text-align: center;

    overflow: hidden;

    box-shadow:
        0 25px 65px rgba(34,21,18,0.40),
        inset 0 1px 0 rgba(217,196,185,0.07);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}


.upload-card::before {
    content: "";

    position: absolute;

    width: 400px;
    height: 200px;

    left: 50%;
    top: -100px;

    transform: translateX(-50%);

    background:
        radial-gradient(
            ellipse,
            rgba(180,158,148,0.16),
            transparent 70%
        );

    pointer-events: none;
}


.upload-card:hover {
    transform: translateY(-4px);

    border-color: rgba(217,196,185,0.55);

    box-shadow:
        0 30px 75px rgba(34,21,18,0.52),
        0 0 35px rgba(114,70,59,0.18);
}


.upload-icon {
    width: 68px;
    height: 68px;

    margin: 0 auto 22px;

    display: flex;

    align-items: center;
    justify-content: center;

    border: 1px solid var(--border-strong);

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.48),
            rgba(34,21,18,0.55)
        );

    color: var(--silver);

    font-size: 28px;

    box-shadow:
        0 15px 35px rgba(34,21,18,0.40),
        inset 0 1px 0 rgba(217,196,185,0.10);
}


.upload-main-title {
    color: var(--silver);

    font-size: 22px;

    font-weight: 700;

    margin-bottom: 9px;
}


.upload-description {
    color: var(--tuscany);

    font-size: 14px;

    line-height: 1.6;

    opacity: 0.78;
}


.upload-formats {
    margin-top: 22px;

    color: var(--metallic);

    font-size: 10px;

    font-weight: 650;

    letter-spacing: 2px;
}


/* =========================================================
   REAL STREAMLIT FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    max-width: 860px;

    margin: -8px auto 45px;
}


[data-testid="stFileUploader"] section {
    border: 1px solid rgba(217,196,185,0.18) !important;

    border-radius: 18px !important;

    background:
        rgba(114,70,59,0.18) !important;

    padding: 14px 18px !important;

    box-shadow:
        inset 0 1px 0 rgba(217,196,185,0.04);
}


[data-testid="stFileUploader"] section:hover {
    border-color: rgba(217,196,185,0.35) !important;

    background:
        rgba(114,70,59,0.25) !important;
}


[data-testid="stFileUploader"] button {
    border: 1px solid rgba(217,196,185,0.22) !important;

    border-radius: 11px !important;

    background:
        #72463B !important;

    color: #D9C4B9 !important;

    font-weight: 650 !important;
}


[data-testid="stFileUploader"] button:hover {
    background:
        #8B5749 !important;

    border-color:
        rgba(217,196,185,0.40) !important;
}


/* =========================================================
   MAIN STREAMLIT BUTTON
   ========================================================= */

.stButton > button {
    height: 58px;

    border-radius: 17px;

    border: 1px solid rgba(217,196,185,0.30);

    color: #D9C4B9 !important;

    font-size: 15px;

    font-weight: 700;

    letter-spacing: 0.4px;

    background:
        linear-gradient(
            135deg,
            #72463B,
            #4E2C25
        ) !important;

    box-shadow:
        0 15px 35px rgba(34,21,18,0.50),
        inset 0 1px 0 rgba(217,196,185,0.12);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}


.stButton > button:hover {
    transform: translateY(-3px);

    border-color:
        rgba(217,196,185,0.50);

    background:
        linear-gradient(
            135deg,
            #8B5749,
            #72463B
        ) !important;

    box-shadow:
        0 20px 45px rgba(34,21,18,0.60),
        0 0 25px rgba(114,70,59,0.20);
}


/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"] {
    border: 1px solid rgba(217,196,185,0.16) !important;

    border-radius: 17px !important;

    background:
        rgba(114,70,59,0.13) !important;

    overflow: hidden;
}


/* =========================================================
   IMAGE
   ========================================================= */

[data-testid="stImage"] {
    border-radius: 22px;

    overflow: hidden;

    box-shadow:
        0 20px 55px rgba(34,21,18,0.38);
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #221512;
}

::-webkit-scrollbar-thumb {
    background: #72463B;

    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: #8B5749;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    .hero-section {
        padding: 75px 15px 80px;
    }

    .hero-title {
        letter-spacing: -3px;
    }

    .currency-list {
        grid-template-columns:
            repeat(3, 1fr);
    }

    .workflow-grid {
        flex-direction: column;
    }

    .upload-title,
    .workflow-title {
        font-size: 30px;
    }
}


@media (max-width: 450px) {

    .currency-list {
        grid-template-columns:
            repeat(2, 1fr);
    }

}


/* =========================================================
   FOOTER
   ========================================================= */

.site-footer {
    margin-top: 120px;

    padding: 40px 20px 25px;

    border-top:
        1px solid rgba(217,196,185,0.12);

    text-align: center;
}


.footer-brand {
    color: var(--silver);

    font-size: 13px;

    font-weight: 700;

    letter-spacing: 2px;

    margin-bottom: 10px;
}


.footer-text {
    color: var(--tuscany);

    font-size: 11px;

    line-height: 1.6;
}


.footer-tech {
    margin-top: 18px;

    color: var(--metallic);

    font-size: 9px;

    font-weight: 650;

    letter-spacing: 1.8px;
}

</style>
""")


st.html("""
<style>

/* ================================
   YOUR BROWN COLOR THEME
   ================================ */

:root {
    --chocolate: #221512;
    --bole: #72463B;
    --pale-silver: #D9C4B9;
    --tuscany: #B49E94;
    --rocket-metallic: #8B8282;
}

/* Main background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:
        radial-gradient(
            circle at 50% -10%,
            rgba(114, 70, 59, 0.35),
            transparent 45%
        ),
        linear-gradient(
            180deg,
            #221512 0%,
            #1c110e 100%
        ) !important;
}

/* Header */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #221512 !important;
}

/* Main text */
.stApp {
    color: #D9C4B9;
}

/* Buttons */
.stButton > button {
    background: #72463B !important;
    color: #D9C4B9 !important;

    border: 1px solid #B49E94 !important;
    border-radius: 14px !important;

    font-weight: 650;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.30);
}

.stButton > button:hover {
    background: #8B5749 !important;
    border-color: #D9C4B9 !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background: rgba(114,70,59,0.20) !important;

    border: 1px solid rgba(180,158,148,0.35) !important;

    border-radius: 18px !important;
}

/* File uploader button */
[data-testid="stFileUploader"] button {
    background: #72463B !important;
    color: #D9C4B9 !important;

    border: 1px solid #B49E94 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(114,70,59,0.16) !important;

    border: 1px solid rgba(180,158,148,0.25) !important;

    border-radius: 16px !important;
}

/* Images */
[data-testid="stImage"] {
    border-radius: 18px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #221512;
}

::-webkit-scrollbar-thumb {
    background: #72463B;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #B49E94;
}

</style>
""")

st.html("""
<style>

/* ================================
   GLOBAL WEBSITE FOUNDATION
   ================================ */

.stApp {
    background:
        radial-gradient(
            circle at 50% -15%,
            rgba(180,158,148,0.12),
            transparent 42%
        ),
        linear-gradient(
            180deg,
            #221512 0%,
            #2a1a16 45%,
            #221512 100%
        );
}

/* Main content width */
.block-container {
    max-width: 1180px;
    padding-top: 0rem;
    padding-bottom: 5rem;
}

/* Remove unnecessary Streamlit spacing */
.element-container {
    margin-bottom: 0.35rem;
}

/* Global typography */
html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color: #D9C4B9;
}

/* ================================
   STREAMLIT BUTTON
   ================================ */

.stButton > button {
    height: 56px;
    border-radius: 16px;

    border: 1px solid rgba(217,196,185,0.28);

    font-size: 15px;
    font-weight: 650;
    letter-spacing: 0.3px;

    color: #D9C4B9;

    background:
        linear-gradient(
            135deg,
            #72463B,
            #5B352D
        );

    box-shadow:
        0 12px 30px rgba(34,21,18,0.45),
        inset 0 1px 0 rgba(217,196,185,0.12);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease,
        background 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    border-color: rgba(217,196,185,0.45);

    background:
        linear-gradient(
            135deg,
            #8B5749,
            #72463B
        );

    box-shadow:
        0 16px 35px rgba(34,21,18,0.55),
        inset 0 1px 0 rgba(217,196,185,0.16);
}

/* ================================
   FILE UPLOADER
   ================================ */

[data-testid="stFileUploader"] {
    max-width: 820px;
    margin: -8px auto 45px auto;
}

[data-testid="stFileUploader"] section {
    border: 1px solid rgba(180,158,148,0.20);
    border-radius: 16px;
    background: rgba(114,70,59,0.16);
    padding: 12px 18px;
    transition:
        border-color 0.25s ease,
        background 0.25s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.04);
}

/* Make the uploader button match the main design */
[data-testid="stFileUploader"] button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.06);
    font-weight: 600;
}

[data-testid="stFileUploader"] button:hover {
    border-color: rgba(255,255,255,0.20);
    background: rgba(255,255,255,0.10);
}

[data-testid="stFileUploader"] section {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    background: rgba(114,70,59,0.18);
    padding: 14px 18px;
    transition:
        border-color 0.25s ease,
        background 0.25s ease,
        transform 0.25s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.04);
    transform: translateY(-1px);
}


/* Hide Streamlit's unnecessary uploader text */
[data-testid="stFileUploaderDropzoneInstructions"] {
    opacity: 0.55;
}

/* ================================
   EXPANDER
   ================================ */

[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    background: rgba(255,255,255,0.025);
    overflow: hidden;
}

/* ================================
   IMAGE DISPLAY
   ================================ */

[data-testid="stImage"] {
    border-radius: 18px;
    overflow: hidden;
}

/* ================================
   SCROLLBAR
   ================================ */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.25);
}

/* ================================
   MOBILE
   ================================ */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

/* ================================
   STREAMLIT CHROME
   ================================ */

/* Hide the default top-right menu */
#MainMenu {
    visibility: hidden;
}

/* Hide Streamlit's default footer */
footer {
    visibility: hidden;
}

/* Remove the extra header space */
header {
    background: transparent !important;
}

/* Keep the page clean while scrolling */
[data-testid="stHeader"] {
    background: transparent;
}

</style>
""")
st.html("""
<style>

.hero-section {
    position: relative;
    text-align: center;
    padding: 105px 20px 85px 20px;
    overflow: hidden;
}

/* Soft glow behind the headline */
.hero-section::before {
    content: "";
    position: absolute;
    width: 520px;
    height: 300px;
    left: 50%;
    top: 15px;
    transform: translateX(-50%);
    background: radial-gradient(
        ellipse,
        rgba(255,255,255,0.09),
        transparent 70%
    );
    filter: blur(10px);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 8px 14px;
    margin-bottom: 28px;

    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 999px;

    background: rgba(114,70,59,0.25);

    font-size: 11px;
    font-weight: 650;
    letter-spacing: 2.5px;
    opacity: 0.65;
}

.hero-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: rgba(255,255,255,0.65);
}

.hero-title {
    font-size: clamp(48px, 7vw, 82px);
    font-weight: 760;
    line-height: 0.98;
    letter-spacing: -4px;
    margin: 0 auto 28px auto;
    max-width: 850px;
}

.hero-title span {
    opacity: 0.42;
}

.hero-description {
    max-width: 610px;
    margin: 0 auto;

    font-size: 17px;
    line-height: 1.7;
    letter-spacing: 0.1px;

    opacity: 0.52;
}

.hero-meta {
    margin-top: 34px;

    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;

    font-size: 11px;
    letter-spacing: 1.5px;
    opacity: 0.32;
}

.hero-meta-line {
    width: 28px;
    height: 1px;
    background: rgba(255,255,255,0.25);
}

</style>

<div class="hero-section">

    <div class="hero-content">

        <div class="hero-badge">
            <span class="hero-dot"></span>
            INDIAN CURRENCY AI
        </div>

        <div class="hero-title">
            Know your<br>
            <span>currency.</span>
        </div>

        <div class="hero-description">
            Identify Indian currency denominations with
            AI-powered YOLOv8 detection — fast, simple,
            and built for everyday use.
        </div>

        <div class="hero-meta">
            <span class="hero-meta-line"></span>
            ₹10 — ₹500
            <span class="hero-meta-line"></span>
        </div>

    </div>

</div>
""")

st.html("""
<style>

.currency-section {
    max-width: 900px;
    margin: 0 auto 90px auto;
}

.currency-header {
    text-align: center;
    margin-bottom: 30px;
}

.currency-label {
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 3px;
    opacity: 0.32;
    margin-bottom: 11px;
}

.currency-title {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
}

.currency-subtitle {
    font-size: 13px;
    opacity: 0.38;
}

.currency-list {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
}

.currency-chip {
    min-height: 72px;

    display: flex;
    align-items: center;
    justify-content: center;

    border: 1px solid rgba(180,158,148,0.18);
    border-radius: 15px;

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.28),
            rgba(114,70,59,0.10)
        );

    color: #D9C4B9;

    font-size: 17px;
    font-weight: 650;

    box-shadow:
        inset 0 1px 0 rgba(217,196,185,0.05);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease,
        background 0.2s ease,
        box-shadow 0.2s ease;
}

.currency-chip:hover {
    transform: translateY(-3px);

    border-color: rgba(217,196,185,0.35);

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.45),
            rgba(114,70,59,0.18)
        );

    box-shadow:
        0 12px 28px rgba(34,21,18,0.35),
        inset 0 1px 0 rgba(217,196,185,0.08);
}

@media (max-width: 768px) {

    .currency-list {
        grid-template-columns: repeat(3, 1fr);
    }

}

@media (max-width: 450px) {

    .currency-list {
        grid-template-columns: repeat(2, 1fr);
    }

}

</style>

<div class="currency-section">

    <div class="currency-header">

        <div class="currency-label">
            SUPPORTED DENOMINATIONS
        </div>

        <div class="currency-title">
            Indian notes, recognized by AI.
        </div>

        <div class="currency-subtitle">
            The current model supports six Indian currency denominations.
        </div>

    </div>

    <div class="currency-list">

        <div class="currency-chip">₹10</div>
        <div class="currency-chip">₹20</div>
        <div class="currency-chip">₹50</div>
        <div class="currency-chip">₹100</div>
        <div class="currency-chip">₹200</div>
        <div class="currency-chip">₹500</div>

    </div>

</div>
""")

st.html("""
<style>

.tech-strip {
    max-width: 900px;
    margin: -30px auto 80px auto;

    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;

    flex-wrap: wrap;
}

.tech-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 9px 14px;

    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 999px;

    background: rgba(255,255,255,0.018);

    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.3px;

    opacity: 0.42;
}

.tech-dot {
    width: 4px;
    height: 4px;

    border-radius: 50%;

    background: rgba(255,255,255,0.55);
}

.tech-separator {
    width: 1px;
    height: 18px;

    background: rgba(255,255,255,0.08);
}

</style>

<div class="tech-strip">

    <div class="tech-item">
        <span class="tech-dot"></span>
        YOLOv8
    </div>

    <div class="tech-separator"></div>

    <div class="tech-item">
        <span class="tech-dot"></span>
        COMPUTER VISION
    </div>

    <div class="tech-separator"></div>

    <div class="tech-item">
        <span class="tech-dot"></span>
        STREAMLIT
    </div>

</div>
""")

st.html("""
<style>

.section-divider {
    max-width: 900px;
    height: 1px;

    margin: 0 auto 80px auto;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.09),
            transparent
        );
}

</style>

<div class="section-divider"></div>
""")

st.html("""
<style>

.workflow-section {
    padding: 20px 0 85px 0;
}

.workflow-header {
    text-align: center;
    margin-bottom: 38px;
}

.workflow-eyebrow {
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 3px;
    opacity: 0.32;
    margin-bottom: 12px;
}

.workflow-title {
    font-size: 34px;
    font-weight: 720;
    letter-spacing: -1px;
    margin-bottom: 10px;
}

.workflow-subtitle {
    font-size: 14px;
    opacity: 0.42;
}

.workflow-grid {
    display: flex;
    gap: 16px;
}

.workflow-card {
    position: relative;
    flex: 1;
    min-height: 205px;

    padding: 27px;

    border: 1px solid rgba(180,158,148,0.18);
    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.24),
            rgba(34,21,18,0.42)
        );

    box-shadow:
        0 18px 40px rgba(34,21,18,0.25),
        inset 0 1px 0 rgba(217,196,185,0.05);

    overflow: hidden;

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        background 0.25s ease,
        box-shadow 0.25s ease;
}

.workflow-card:hover {
    transform: translateY(-5px);

    border-color: rgba(217,196,185,0.32);

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.36),
            rgba(34,21,18,0.48)
        );

    box-shadow:
        0 22px 48px rgba(34,21,18,0.38),
        inset 0 1px 0 rgba(217,196,185,0.08);
}

.workflow-number {
    font-size: 11px;
    letter-spacing: 2px;
    opacity: 0.30;
    margin-bottom: 35px;
}

.workflow-icon {
    position: absolute;
    top: 25px;
    right: 25px;

    width: 38px;
    height: 38px;

    display: flex;
    align-items: center;
    justify-content: center;

    border: 1px solid rgba(180,158,148,0.20);
    border-radius: 11px;

    background: rgba(114,70,59,0.22);

    font-size: 16px;
    opacity: 0.65;
}

.workflow-card-title {
    font-size: 20px;
    font-weight: 650;
    margin-bottom: 9px;
}

.workflow-card-text {
    font-size: 13px;
    line-height: 1.6;
    max-width: 270px;
    opacity: 0.43;
}

@media (max-width: 768px) {

    .workflow-grid {
        flex-direction: column;
    }

}

</style>

<div class="workflow-section">

    <div class="workflow-header">

        <div class="workflow-eyebrow">
            HOW IT WORKS
        </div>

        <div class="workflow-title">
            From image to insight.
        </div>

        <div class="workflow-subtitle">
            Three simple steps to identify your currency note.
        </div>

    </div>

    <div class="workflow-grid">

        <div class="workflow-card">

            <div class="workflow-number">
                01 / UPLOAD
            </div>

            <div class="workflow-icon">
                ↑
            </div>

            <div class="workflow-card-title">
                Upload
            </div>

            <div class="workflow-card-text">
                Choose a clear image of an Indian currency note
                from your device.
            </div>

        </div>


        <div class="workflow-card">

            <div class="workflow-number">
                02 / ANALYZE
            </div>

            <div class="workflow-icon">
                ◉
            </div>

            <div class="workflow-card-title">
                Analyze
            </div>

            <div class="workflow-card-text">
                Our YOLOv8 model processes the image and
                identifies the denomination.
            </div>

        </div>


        <div class="workflow-card">

            <div class="workflow-number">
                03 / RESULT
            </div>

            <div class="workflow-icon">
                ✓
            </div>

            <div class="workflow-card-title">
                Result
            </div>

            <div class="workflow-card-text">
                View the detected denomination together with
                the model's confidence score.
            </div>

        </div>

    </div>

</div>
""")


st.html("""
<style>

.upload-section {
    max-width: 820px;
    margin: 0 auto 18px auto;
}

.upload-heading {
    text-align: center;
    margin-bottom: 25px;
}

.upload-eyebrow {
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 3px;
    opacity: 0.32;
    margin-bottom: 11px;
}

.upload-title {
    font-size: 34px;
    font-weight: 720;
    letter-spacing: -1px;
    margin-bottom: 9px;
}

.upload-subtitle {
    font-size: 14px;
    opacity: 0.42;
}

/* Main upload surface */
 .upload-card {
    position: relative;

    padding: 48px 30px 42px 30px;

    border: 1px dashed rgba(180,158,148,0.30);
    border-radius: 24px;

    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(114,70,59,0.30),
            transparent 60%
        ),
        rgba(34,21,18,0.55);

    text-align: center;

    overflow: hidden;

    box-shadow:
        0 20px 50px rgba(34,21,18,0.28),
        inset 0 1px 0 rgba(217,196,185,0.05);

    transition:
        border-color 0.25s ease,
        background 0.25s ease,
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

.upload-card:hover {
    border-color: rgba(217,196,185,0.50);

    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(114,70,59,0.42),
            transparent 62%
        ),
        rgba(34,21,18,0.62);

    transform: translateY(-2px);

    box-shadow:
        0 24px 60px rgba(34,21,18,0.38),
        inset 0 1px 0 rgba(217,196,185,0.08);
}

.upload-icon {
    width: 58px;
    height: 58px;

    margin: 0 auto 20px auto;

    display: flex;
    align-items: center;
    justify-content: center;

    border: 1px solid rgba(180,158,148,0.24);
    border-radius: 17px;

    background: rgba(114,70,59,0.25);
    color: #D9C4B9;

    font-size: 25px;
    font-weight: 300;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.15),
        inset 0 1px 0 rgba(255,255,255,0.06);
}

.upload-main-title {
    font-size: 21px;
    font-weight: 650;
    margin-bottom: 9px;
}

.upload-description {
    font-size: 14px;
    line-height: 1.6;
    opacity: 0.43;
}

.upload-formats {
    margin-top: 22px;

    font-size: 10px;
    letter-spacing: 1.5px;
    opacity: 0.28;
}

</style>

<div class="upload-section">

    <div class="upload-heading">

        <div class="upload-eyebrow">
            TRY IT YOURSELF
        </div>

        <div class="upload-title">
            Analyze a currency note.
        </div>

        <div class="upload-subtitle">
            Upload an image and let the model identify its denomination.
        </div>

    </div>

    <div class="upload-card">

        <div class="upload-icon">
            ↑
        </div>

        <div class="upload-main-title">
            Drop your currency note here
        </div>

        <div class="upload-description">
            Drag and drop your image here, or use the uploader below.
        </div>

        <div class="upload-formats">
            JPG &nbsp;·&nbsp; JPEG &nbsp;·&nbsp; PNG
        </div>

    </div>

</div>
""")

uploaded_file = st.file_uploader(
    "Select your currency note",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Supported formats: JPG, JPEG and PNG"
)

if uploaded_file is not None:

    st.html("""
<div style="
    text-align: center;
    margin: 18px 0;
    font-size: 14px;
    opacity: 0.6;
">
    ✓ Image ready for AI analysis
</div>
""")

    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.html("""
<div style="
    margin-top: 25px;
    margin-bottom: 15px;
">

    <div style="
        font-size: 12px;
        letter-spacing: 3px;
        font-weight: 600;
        opacity: 0.45;
        margin-bottom: 8px;
    ">
        YOUR UPLOAD
    </div>

    <div style="
        font-size: 24px;
        font-weight: 700;
    ">
        Uploaded Currency Note
    </div>

</div>
""")

    st.html("""
    <div style="
        font-size: 13px;
        letter-spacing: 1px;
        opacity: 0.45;
        margin: 10px 0 8px 2px;
    ">
        ORIGINAL IMAGE
    </div>
    """)

    st.image(
        image,
        use_container_width=True
    )

    with st.expander("File Details"):
        st.html(f"""
    <div style="
        padding: 8px 4px 4px 4px;
    ">

        <div style="
            display: grid;
            grid-template-columns: 130px 1fr;
            gap: 12px 20px;
            font-size: 13px;
        ">

            <div style="opacity:0.35;">FILE NAME</div>
            <div style="opacity:0.7; font-weight:500;">
                {uploaded_file.name}
            </div>

            <div style="opacity:0.35;">FILE TYPE</div>
            <div style="opacity:0.7; font-weight:500;">
                {uploaded_file.type}
            </div>

            <div style="opacity:0.35;">FILE SIZE</div>
            <div style="opacity:0.7; font-weight:500;">
                {round(uploaded_file.size / 1024, 2)} KB
            </div>

        </div>

    </div>
    """)

st.html("""
<style>

.scan-action {
    max-width: 820px;
    margin: 5px auto 35px auto;
    text-align: center;
}

.scan-label {
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 2.5px;
    opacity: 0.28;
    margin-bottom: 12px;
}

.scan-button-wrap {
    padding: 4px;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.10),
            rgba(255,255,255,0.025)
        );

    border: 1px solid rgba(255,255,255,0.07);
}

.scan-hint {
    margin-top: 12px;

    font-size: 11px;
    letter-spacing: 0.4px;

    opacity: 0.25;
}

</style>

<div class="scan-action">

    <div class="scan-label">
        READY TO ANALYZE
    </div>

    <div class="scan-button-wrap">
</div>

<div class="scan-hint">
    AI-powered denomination detection
</div>

</div>
""")

analyze = st.button(
    "✦ Analyze Currency  →",
    use_container_width=True,
    type="primary"
)


if uploaded_file is not None:
    if analyze:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        denomination, confidence, result = detect_currency(image)

        real_fake, real_fake_confidence = predict_real_fake(image)

        if denomination is not None:

            confidence_percent = confidence * 100
            if real_fake.lower() == "real":
                voice_text = f"The scan is completed and The detected currency note is {denomination} rupees. And there appears No Red Flags here! and hence the ccurrency is genuine"
            else:
                voice_text = f"The scan is completed and The detected currency note is {denomination} rupees. And there's definitely a Red Flag here! so its a fake currency"

            tts = gTTS(text=voice_text, lang="en")
            tts.save("currency_voice.mp3")

            import base64

            with open("currency_voice.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()

            audio_base64 = base64.b64encode(audio_bytes).decode()

            components.html(
    f"""
    <style>
        body {{
            margin: 0;
            background: transparent;
            font-family: Arial, sans-serif;
        }}

        .voice-section {{
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .voice-wave {{
            width: 720px;
            max-width: 95%;
            height: 95px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 5px;
        }}

        .bar {{
            width: 4px;
            border-radius: 10px;
            background: #B49E94;
            opacity: 0.8;
            transition: height 0.2s ease;
        }}

        /* STILL WAVE */
        .bar:nth-child(1)  {{ height: 10px; }}
        .bar:nth-child(2)  {{ height: 15px; }}
        .bar:nth-child(3)  {{ height: 22px; }}
        .bar:nth-child(4)  {{ height: 30px; }}
        .bar:nth-child(5)  {{ height: 40px; }}
        .bar:nth-child(6)  {{ height: 52px; }}
        .bar:nth-child(7)  {{ height: 64px; }}
        .bar:nth-child(8)  {{ height: 75px; }}
        .bar:nth-child(9)  {{ height: 84px; }}
        .bar:nth-child(10) {{ height: 75px; }}
        .bar:nth-child(11) {{ height: 64px; }}
        .bar:nth-child(12) {{ height: 52px; }}
        .bar:nth-child(13) {{ height: 40px; }}
        .bar:nth-child(14) {{ height: 30px; }}
        .bar:nth-child(15) {{ height: 22px; }}
        .bar:nth-child(16) {{ height: 15px; }}
        .bar:nth-child(17) {{ height: 10px; }}
        .bar:nth-child(18) {{ height: 15px; }}
        .bar:nth-child(19) {{ height: 22px; }}
        .bar:nth-child(20) {{ height: 30px; }}
        .bar:nth-child(21) {{ height: 40px; }}
        .bar:nth-child(22) {{ height: 52px; }}
        .bar:nth-child(23) {{ height: 64px; }}
        .bar:nth-child(24) {{ height: 75px; }}
        .bar:nth-child(25) {{ height: 84px; }}
        .bar:nth-child(26) {{ height: 75px; }}
        .bar:nth-child(27) {{ height: 64px; }}
        .bar:nth-child(28) {{ height: 52px; }}
        .bar:nth-child(29) {{ height: 40px; }}
        .bar:nth-child(30) {{ height: 30px; }}
        .bar:nth-child(31) {{ height: 22px; }}
        .bar:nth-child(32) {{ height: 15px; }}
        .bar:nth-child(33) {{ height: 10px; }}

        /* MOVING WAVE */
        .playing .bar {{
            animation: wave 0.75s ease-in-out infinite alternate;
        }}

        .playing .bar:nth-child(1)  {{ animation-delay: 0.00s; }}
        .playing .bar:nth-child(2)  {{ animation-delay: 0.05s; }}
        .playing .bar:nth-child(3)  {{ animation-delay: 0.10s; }}
        .playing .bar:nth-child(4)  {{ animation-delay: 0.15s; }}
        .playing .bar:nth-child(5)  {{ animation-delay: 0.20s; }}
        .playing .bar:nth-child(6)  {{ animation-delay: 0.25s; }}
        .playing .bar:nth-child(7)  {{ animation-delay: 0.30s; }}
        .playing .bar:nth-child(8)  {{ animation-delay: 0.35s; }}
        .playing .bar:nth-child(9)  {{ animation-delay: 0.40s; }}
        .playing .bar:nth-child(10) {{ animation-delay: 0.35s; }}
        .playing .bar:nth-child(11) {{ animation-delay: 0.30s; }}
        .playing .bar:nth-child(12) {{ animation-delay: 0.25s; }}
        .playing .bar:nth-child(13) {{ animation-delay: 0.20s; }}
        .playing .bar:nth-child(14) {{ animation-delay: 0.15s; }}
        .playing .bar:nth-child(15) {{ animation-delay: 0.10s; }}
        .playing .bar:nth-child(16) {{ animation-delay: 0.05s; }}

        @keyframes wave {{
            0% {{
                transform: scaleY(0.45);
                opacity: 0.55;
            }}

            50% {{
                transform: scaleY(1.15);
                opacity: 1;
            }}

            100% {{
                transform: scaleY(0.65);
                opacity: 0.7;
            }}
        }}

        .voice-button {{
            margin-top: 5px;
            padding: 9px 25px;
            border: 1px solid #B49E94;
            border-radius: 25px;
            background: #221512;
            color: #D9C4B9;
            font-size: 14px;
            cursor: pointer;
        }}

        .voice-button:hover {{
            background: #72463B;
            color: white;
        }}
    </style>

    <div class="voice-section">

        <div id="wave" class="voice-wave">

            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>

        </div>

        <button class="voice-button" onclick="playVoice()">
            ▶ Play Voice
        </button>

        <audio id="currencyAudio">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>

    </div>

    <script>
        const audio = document.getElementById("currencyAudio");
        const wave = document.getElementById("wave");

        function playVoice() {{
            audio.currentTime = 0;
            audio.play();
        }}

        audio.addEventListener("play", function() {{
            wave.classList.add("playing");
        }});

        audio.addEventListener("ended", function() {{
            wave.classList.remove("playing");
        }});

        audio.addEventListener("pause", function() {{
            wave.classList.remove("playing");
        }});

        // Try automatic playback
        audio.play().catch(function() {{
            // Browser may block autoplay.
            // The Play Voice button can still be used.
        }});
    </script>
    """,
    height=145,
)

            # Result card
            # Result card
            st.html(f"""
            <style>

            .result-section {{
                margin: 55px auto 35px auto;
                max-width: 820px;
            }}

            .result-card {{
                position: relative;
                text-align: center;

                padding: 48px 30px 42px 30px;

                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 26px;

                background:
                    radial-gradient(
                        circle at 50% 0%,
                        rgba(255,255,255,0.075),
                        transparent 58%
                    ),
                    rgba(255,255,255,0.025);

                box-shadow:
                    0 25px 70px rgba(0,0,0,0.18),
                    inset 0 1px 0 rgba(255,255,255,0.04);
            }}

            .result-eyebrow {{
                font-size: 10px;
                font-weight: 650;
                letter-spacing: 3px;
                opacity: 0.35;
                margin-bottom: 20px;
            }}

            .result-denomination {{
                font-size: clamp(64px, 9vw, 96px);
                font-weight: 780;
                line-height: 0.95;
                letter-spacing: -5px;
                margin-bottom: 25px;
            }}

            .result-label {{
                font-size: 13px;
                pacity: 0.40;
                margin-bottom: 7px;
            }}

            .result-confidence {{
                font-size: 25px;
                font-weight: 680;
            }}

            .result-divider {{
                width: 45px;
                height: 1px;
                margin: 27px auto 0 auto;
                background: rgba(255,255,255,0.12);
            }}

            </style>

            <div class="result-section">

                <div class="result-card">

                    <div class="result-eyebrow">
                        DETECTION RESULT
                    </div>

                    <div class="result-denomination">
                        ₹{denomination}
                    </div>

                    <div class="result-label">
                        AI CONFIDENCE
                    </div>

                    <div class="result-confidence">
                        {confidence_percent:.2f}%
                    </div>

                    <div class="result-divider"></div>

                </div>

            </div>
            """)

            st.markdown(
    f'<div style="text-align:center; margin-top:18px; padding:18px; border-radius:16px; background:rgba(180,158,148,0.10); border:1px solid rgba(180,158,148,0.25);"><div style="font-size:11px; letter-spacing:2px; opacity:0.55; margin-bottom:8px;">AUTHENTICITY</div><div style="font-size:28px; font-weight:700;">{real_fake.upper()}</div><div style="font-size:13px; opacity:0.55; margin-top:5px;">AI Confidence: {real_fake_confidence:.2f}%</div></div>',
    unsafe_allow_html=True
)
            
            # Confidence status
            # Confidence status
            if confidence_percent >= 90:
                status_text = "HIGH CONFIDENCE"
                status_description = "The model is highly confident in this detection."

            elif confidence_percent >= 70:
                status_text = "GOOD CONFIDENCE"
                status_description = "The model has a good level of confidence in this detection."

            else:
                status_text = "LOW CONFIDENCE"
                status_description = "Consider uploading a clearer image for a better result."

            confidence_width = max(0, min(100, confidence_percent))

            st.html(f"""
            <style>

            .confidence-panel {{
                max-width: 820px;
                margin: 18px auto 45px auto;

                padding: 24px 28px;

                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;

                background:
                    linear-gradient(
                        145deg,
                        rgba(255,255,255,0.035),
                        rgba(255,255,255,0.015)
                    );
                }}

            .confidence-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;

                margin-bottom: 13px;
            }}

            .confidence-label {{
                font-size: 10px;
                font-weight: 650;
                letter-spacing: 2.5px;
                opacity: 0.38;
            }}

            .confidence-value {{
                font-size: 15px;
                font-weight: 650;
                opacity: 0.75;
            }}

            .confidence-track {{
                width: 100%;
                height: 5px;

                border-radius: 999px;

                background: rgba(255,255,255,0.07);

                overflow: hidden;
            }}

            .confidence-fill {{
                width: {confidence_width:.2f}%;

                height: 100%;

                border-radius: 999px;

                background:
                    linear-gradient(
                        90deg,
                        rgba(255,255,255,0.35),
                        rgba(255,255,255,0.85)
                    );

                box-shadow:
                    0 0 18px rgba(255,255,255,0.12);
            }}

            .confidence-status {{
                margin-top: 15px;

                font-size: 11px;
                font-weight: 650;
                letter-spacing: 1.8px;

                opacity: 0.55;
            }}

            .confidence-description {{
                margin-top: 5px;

                font-size: 12px;
                line-height: 1.5;

                opacity: 0.35;
            }}

            </style>

            <div class="confidence-panel">

                <div class="confidence-header">

                    <div class="confidence-label">
                        MODEL CONFIDENCE
                    </div>

                <div class="confidence-value">
                    {confidence_percent:.2f}%
                </div>

            </div>

            <div class="confidence-track">
            <div class="confidence-fill"></div>
            </div>

    <div class="confidence-status">
        {status_text}
    </div>

    <div class="confidence-description">
        {status_description}
    </div>

</div>
""")

            # Result explanation
            st.html(f"""
<style>

.result-explanation {{
    max-width: 820px;
    margin: 0 auto 65px auto;
}}

.explanation-card {{
    display: flex;
    align-items: center;
    gap: 22px;

    padding: 22px 25px;

    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;

    background: rgba(255,255,255,0.018);
}}

.explanation-icon {{
    flex-shrink: 0;

    width: 42px;
    height: 42px;

    display: flex;
    align-items: center;
    justify-content: center;

    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 13px;

    background: rgba(255,255,255,0.035);

    font-size: 16px;
    opacity: 0.65;
}}

.explanation-content {{
    text-align: left;
}}

.explanation-title {{
    font-size: 13px;
    font-weight: 650;
    margin-bottom: 5px;
    opacity: 0.65;
}}

.explanation-text {{
    font-size: 12px;
    line-height: 1.55;
    opacity: 0.35;
}}

@media (max-width: 600px) {{

    .explanation-card {{
        align-items: flex-start;
    }}

}}

</style>

<div class="result-explanation">

    <div class="explanation-card">

        <div class="explanation-icon">
            ✓
        </div>

        <div class="explanation-content">

            <div class="explanation-title">
                What the result means
            </div>

            <div class="explanation-text">
                The model identified this note as ₹{denomination}
                based on the visual features detected in the uploaded image.
                The confidence score indicates how strongly the model supports
                this prediction.
            </div>

        </div>

    </div>

</div>
""")

            # Detection preview
            # Detection preview
            st.html("""
            <style>

            .preview-section {
                max-width: 900px;
                margin: 70px auto 25px auto;
            }

            .preview-header {
                text-align: center;
                margin-bottom: 28px;
            }

            .preview-eyebrow {
                font-size: 10px;
                font-weight: 650;
                letter-spacing: 3px;
                opacity: 0.32;
                margin-bottom: 11px;
            }

            .preview-title {
                font-size: 30px;
                font-weight: 720;
                letter-spacing: -1px;
                margin-bottom: 8px;
            }

            .preview-description {
                font-size: 13px;
                opacity: 0.40;
            }

            .preview-frame {
                position: relative;

                padding: 12px;

                border: 1px solid rgba(255,255,255,0.09);
                border-radius: 24px;

                background:
                    radial-gradient(
                        circle at 50% 0%,
                        rgba(114,70,59,0.32),
                        transparent 55%
                    ),
                rgba(255,255,255,0.018);

                box-shadow:
                        0 25px 70px rgba(0,0,0,0.20),
                        inset 0 1px 0 rgba(255,255,255,0.04);
            }

            .preview-tag {
                position: absolute;
                top: 24px;
                left: 24px;

                padding: 7px 11px;

                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 9px;

                background: rgba(10,10,12,0.72);

                backdrop-filter: blur(10px);

                font-size: 9px;
                font-weight: 650;
                letter-spacing: 1.8px;

                opacity: 0.70;
            }

            .preview-caption {
                text-align: center;

                margin-top: 17px;

                font-size: 11px;
                letter-spacing: 0.8px;

                opacity: 0.28;
            }

            </style>

            <div class="preview-section">

                <div class="preview-header">

                    <div class="preview-eyebrow">
                        AI VISUALIZATION
                    </div>

                    <div class="preview-title">
                        Detection Preview
                    </div>

                    <div class="preview-description">
                        Visual representation of the model's detected currency note.
                    </div>

                </div>

            <div class="preview-frame">

                <div class="preview-tag">
                    YOLOv8 DETECTION
                </div>

            </div>
            """)

            annotated_image = result.plot()

            st.image(
                annotated_image,
                use_container_width=True
            )

            st.html("""
<div style="
    max-width: 900px;
    margin: 0 auto;
    text-align: center;
">
    <div style="
        margin-top: 15px;
        font-size: 11px;
        letter-spacing: 0.7px;
        opacity: 0.28;
    ">
        Bounding box generated by the currency detection model
    </div>
</div>

</div>
""")

        else:

            st.warning("⚠️ No currency note detected.")

            st.info(
                "Please upload a clear image of an Indian currency note "
                "and try again."
            )

    else:
        st.html("""
<div style="
    text-align: center;
    margin: 20px 0 10px 0;
    font-size: 13px;
    opacity: 0.45;
">
    Ready when you are — click <b>Analyze Currency</b> to begin.
</div>
""")

else:
    st.html("""
    <div style="
    text-align: center;
    margin: 20px 0 40px 0;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    background: rgba(255,255,255,0.02);
    font-size: 13px;
    opacity: 0.5;
">
    Upload a currency note above to begin detection.
</div>
""")

st.html("""
<style>

.site-footer {
    margin-top: 110px;
    padding: 35px 20px 25px 20px;

    border-top: 1px solid rgba(255,255,255,0.07);

    text-align: center;
}

.footer-brand {
    font-size: 13px;
    font-weight: 650;
    letter-spacing: 1px;
    opacity: 0.55;
    margin-bottom: 9px;
}

.footer-text {
    font-size: 11px;
    line-height: 1.6;
    opacity: 0.28;
}

.footer-tech {
    margin-top: 17px;
    font-size: 10px;
    letter-spacing: 1.5px;
    opacity: 0.22;
}

</style>

<div class="site-footer">

    <div class="footer-brand">
        INDIAN CURRENCY AI
    </div>

    <div class="footer-text">
        AI-powered currency denomination detection.
    </div>

    <div class="footer-tech">
        YOLOv8 &nbsp;·&nbsp; COMPUTER VISION &nbsp;·&nbsp; STREAMLIT
    </div>

</div>
""")

st.html("""
<style>

/* =====================================================
   FINAL BROWN PALETTE OVERRIDE
   ===================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(114, 70, 59, 0.55),
            transparent 45%
        ),
        linear-gradient(
            180deg,
            #221512 0%,
            #221512 55%,
            #1b100d 100%
        ) !important;
}

/* Main text */
.hero-title,
.currency-title,
.workflow-title,
.upload-title,
.preview-title,
.currency-chip,
.workflow-card-title,
.upload-main-title,
.result-denomination,
.result-confidence {
    color: #D9C4B9 !important;
}

/* Secondary text */
.hero-description,
.hero-meta,
.currency-subtitle,
.workflow-subtitle,
.workflow-card-text,
.upload-subtitle,
.upload-description,
.result-label,
.preview-description {
    color: #B49E94 !important;
}

/* Brown cards */
.currency-chip,
.workflow-card,
.upload-card,
.result-card,
.confidence-panel,
.explanation-card,
.preview-frame {
    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.38),
            rgba(34,21,18,0.72)
        ) !important;

    border-color:
        rgba(180,158,148,0.30) !important;
}

/* Strong brown accent */
.hero-badge,
.tech-item,
.workflow-icon,
.upload-icon,
.explanation-icon,
.preview-tag {
    background: rgba(114,70,59,0.35) !important;
    border-color: rgba(180,158,148,0.35) !important;
}

/* Main button */
.stButton > button {
    background:
        linear-gradient(
            135deg,
            #72463B,
            #8B5749
        ) !important;

    color: #D9C4B9 !important;

    border-color: #B49E94 !important;

    box-shadow:
        0 10px 30px rgba(34,21,18,0.55),
        0 0 20px rgba(114,70,59,0.25);
}

/* Upload area */
[data-testid="stFileUploader"] section {
    background: rgba(114,70,59,0.25) !important;
    border-color: rgba(180,158,148,0.40) !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background: #72463B !important;
    color: #D9C4B9 !important;
    border-color: #B49E94 !important;
}

/* Confidence bar */
.confidence-fill {
    background:
        linear-gradient(
            90deg,
            #72463B,
            #B49E94
        ) !important;

    box-shadow:
        0 0 15px rgba(114,70,59,0.45) !important;
}

/* Dividers */
.section-divider,
.result-divider {
    background:
        linear-gradient(
            90deg,
            transparent,
            #72463B,
            transparent
        ) !important;
}

/* Footer */
.site-footer {
    border-top-color: rgba(180,158,148,0.20) !important;
}

.footer-brand {
    color: #D9C4B9 !important;
}

.footer-text {
    color: #B49E94 !important;
}

.footer-tech {
    color: #8B8282 !important;
}

</style>
""")

st.html("""
<style>

/* =========================================================
   FINAL COLOR THEME
   Indian Currency AI
   ========================================================= */

/* PALETTE
   #221512  Black Chocolate
   #72463B  Bole
   #D9C4B9  Pale Silver
   #B49E94  Tuscany
   #8B8282  Rocket Metallic
*/


/* =========================
   BACKGROUND
   ========================= */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {

    background:
        radial-gradient(
            circle at 50% -10%,
            rgba(114, 70, 59, 0.55),
            transparent 42%
        ),
        radial-gradient(
            circle at 10% 45%,
            rgba(180, 158, 148, 0.08),
            transparent 30%
        ),
        linear-gradient(
            180deg,
            #221512 0%,
            #221512 45%,
            #1b100d 100%
        ) !important;
}


/* =========================
   HEADER
   ========================= */

[data-testid="stHeader"] {
    background: transparent !important;
}


/* =========================
   ALL MAIN TEXT
   ========================= */

.stApp {
    color: #D9C4B9 !important;
}


/* Headings */

.hero-title,
.currency-title,
.workflow-title,
.upload-title,
.preview-title,
.result-denomination,
.result-confidence,
.currency-chip,
.workflow-card-title,
.upload-main-title {
    color: #D9C4B9 !important;
}


/* Secondary text */

.hero-description,
.hero-meta,
.currency-subtitle,
.workflow-subtitle,
.workflow-card-text,
.upload-subtitle,
.upload-description,
.preview-description,
.result-label,
.footer-text {
    color: #B49E94 !important;
}


/* Small labels */

.hero-badge,
.currency-label,
.workflow-eyebrow,
.upload-eyebrow,
.preview-eyebrow,
.upload-formats,
.workflow-number,
.tech-item,
.scan-label {
    color: #B49E94 !important;
}


/* =========================
   HERO
   ========================= */

.hero-section::before {

    background:
        radial-gradient(
            ellipse,
            rgba(114,70,59,0.65),
            rgba(114,70,59,0.18),
            transparent 70%
        ) !important;
}


.hero-title span {
    color: #B49E94 !important;
}


/* =========================
   BROWN GLASS CARDS
   ========================= */

.currency-chip,
.workflow-card,
.upload-card,
.result-card,
.confidence-panel,
.explanation-card,
.preview-frame {

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.42),
            rgba(34,21,18,0.78)
        ) !important;

    border-color:
        rgba(217,196,185,0.24) !important;

    box-shadow:
        0 18px 45px rgba(34,21,18,0.45),
        inset 0 1px 0 rgba(217,196,185,0.06) !important;
}


/* =========================
   CARD HOVER
   ========================= */

.currency-chip:hover,
.workflow-card:hover,
.upload-card:hover {

    background:
        linear-gradient(
            145deg,
            rgba(114,70,59,0.58),
            rgba(34,21,18,0.80)
        ) !important;

    border-color:
        rgba(217,196,185,0.42) !important;
}


/* =========================
   ICON BOXES
   ========================= */

.workflow-icon,
.upload-icon,
.explanation-icon {

    background:
        linear-gradient(
            145deg,
            #72463B,
            #4d2d26
        ) !important;

    border-color:
        rgba(217,196,185,0.30) !important;

    color: #D9C4B9 !important;
}


/* =========================
   BADGE
   ========================= */

.hero-badge {

    background:
        rgba(114,70,59,0.38) !important;

    border-color:
        rgba(217,196,185,0.28) !important;
}


/* =========================
   TECHNOLOGY PILLS
   ========================= */

.tech-item {

    background:
        rgba(114,70,59,0.28) !important;

    border-color:
        rgba(180,158,148,0.25) !important;
}


/* =========================
   UPLOAD AREA
   ========================= */

.upload-card {

    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(114,70,59,0.55),
            transparent 62%
        ),
        linear-gradient(
            145deg,
            rgba(114,70,59,0.30),
            rgba(34,21,18,0.80)
        ) !important;

    border-color:
        rgba(217,196,185,0.40) !important;
}


/* =========================
   REAL STREAMLIT UPLOADER
   ========================= */

[data-testid="stFileUploader"] section {

    background:
        rgba(114,70,59,0.28) !important;

    border:
        1px solid rgba(217,196,185,0.28) !important;

    border-radius:
        18px !important;
}


[data-testid="stFileUploader"] section:hover {

    background:
        rgba(114,70,59,0.40) !important;

    border-color:
        rgba(217,196,185,0.45) !important;
}


[data-testid="stFileUploader"] button {

    background:
        #72463B !important;

    color:
        #D9C4B9 !important;

    border:
        1px solid #B49E94 !important;
}


/* =========================
   ANALYZE BUTTON
   ========================= */

.stButton > button {

    background:
        linear-gradient(
            135deg,
            #72463B 0%,
            #8B5749 100%
        ) !important;

    color:
        #D9C4B9 !important;

    border:
        1px solid #B49E94 !important;

    box-shadow:
        0 12px 35px rgba(34,21,18,0.55),
        0 0 20px rgba(114,70,59,0.25) !important;
}


.stButton > button:hover {

    background:
        linear-gradient(
            135deg,
            #8B5749 0%,
            #72463B 100%
        ) !important;

    color:
        #D9C4B9 !important;

    border-color:
        #D9C4B9 !important;
}


/* =========================
   CONFIDENCE BAR
   ========================= */

.confidence-track {

    background:
        rgba(139,130,130,0.22) !important;
}


.confidence-fill {

    background:
        linear-gradient(
            90deg,
            #72463B,
            #B49E94
        ) !important;

    box-shadow:
        0 0 18px rgba(114,70,59,0.55) !important;
}


/* =========================
   DIVIDERS
   ========================= */

.section-divider,
.result-divider {

    background:
        linear-gradient(
            90deg,
            transparent,
            #72463B,
            #B49E94,
            #72463B,
            transparent
        ) !important;
}


/* =========================
   PREVIEW TAG
   ========================= */

.preview-tag {

    background:
        rgba(34,21,18,0.88) !important;

    border-color:
        rgba(217,196,185,0.28) !important;

    color:
        #D9C4B9 !important;
}


/* =========================
   EXPANDER
   ========================= */

[data-testid="stExpander"] {

    background:
        rgba(114,70,59,0.20) !important;

    border-color:
        rgba(217,196,185,0.22) !important;
}


/* =========================
   FOOTER
   ========================= */

.site-footer {

    border-top-color:
        rgba(180,158,148,0.22) !important;
}


.footer-brand {
    color: #D9C4B9 !important;
}


.footer-text {
    color: #B49E94 !important;
}


.footer-tech {
    color: #8B8282 !important;
}


/* =========================
   SCROLLBAR
   ========================= */

::-webkit-scrollbar-track {
    background: #221512 !important;
}


::-webkit-scrollbar-thumb {
    background: #72463B !important;
}


::-webkit-scrollbar-thumb:hover {
    background: #B49E94 !important;
}

</style>
""")

st.html("""
<style>

/* FORCE BROWN THEME EVERYWHERE */

* {
    --theme-bg: #221512;
    --theme-brown: #72463B;
    --theme-light: #D9C4B9;
    --theme-taupe: #B49E94;
    --theme-gray: #8B8282;
}

.stApp {
    background-color: #221512 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #221512 !important;
}

[data-testid="stHeader"] {
    background: #221512 !important;
}

/* Streamlit text */
.stApp p,
.stApp label,
.stApp span {
    color: #B49E94;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #D9C4B9 !important;
}

/* Buttons */
button {
    background: #72463B !important;
    color: #D9C4B9 !important;
    border-color: #B49E94 !important;
}

/* Inputs / uploader */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background: rgba(114,70,59,0.25) !important;
    border-color: #72463B !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(114,70,59,0.20) !important;
    border-color: rgba(180,158,148,0.30) !important;
}

</style>
""")