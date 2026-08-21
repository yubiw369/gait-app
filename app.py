import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import mediapipe as mp


# =========================================================
# 1. ตั้งค่า MediaPipe
# =========================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# =========================================================
# 2. ตั้งค่าหน้าเว็บ
# =========================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์ท่าเดิน",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 3. CSS สำหรับ UI - Dark Medical AI Dashboard
# =========================================================

st.markdown(
    """
    <style>
    /* ---------- App background ---------- */
    .stApp {
        background:
            radial-gradient(circle at top, #1a4164 0%, #0e2b46 42%, #081a2c 100%);
        color: #f8fafc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hide Streamlit chrome for app-like look ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* ---------- Top navigation ---------- */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        padding: 16px 24px;
        margin-bottom: 24px;
        background: rgba(11, 35, 57, 0.90);
        border: 1px solid rgba(125, 211, 252, 0.12);
        border-radius: 14px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.22);
        backdrop-filter: blur(10px);
    }

    .brand {
        font-size: 24px;
        font-weight: 800;
        color: #eef8ff;
        white-space: nowrap;
    }

    .brand-icon {
        color: #22d3ee;
        margin-right: 8px;
    }

    .nav-items {
        display: flex;
        gap: 32px;
        align-items: center;
        color: #94a3b8;
        font-size: 15px;
    }

    .nav-active {
        color: #ffffff;
        border-bottom: 2px solid #22d3ee;
        padding-bottom: 7px;
        text-shadow: 0 0 14px rgba(34,211,238,0.35);
    }

    .nav-user {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: rgba(148,163,184,0.16);
        border: 1px solid rgba(148,163,184,0.20);
        font-size: 22px;
    }

    /* ---------- Section heading ---------- */
    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #f8fafc;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }

    /* ---------- Medical cards ---------- */
    .med-card {
        background: linear-gradient(
            145deg,
            rgba(31, 73, 108, 0.96),
            rgba(15, 44, 70, 0.98)
        );
        border: 1px solid rgba(103, 232, 249, 0.16);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.28);
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 750;
        color: #f8fafc;
        padding-bottom: 11px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(148,163,184,0.17);
    }

    .score-number {
        font-size: 58px;
        line-height: 1;
        font-weight: 850;
        text-align: center;
        color: #ffffff;
        margin: 12px 0 8px;
    }

    .score-caption {
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
    }

    .screening-status {
        text-align: center;
        font-size: 21px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 10px;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.6;
    }

    /* ---------- Status chips/cards ---------- */
    .status-good, .status-watch, .status-alert {
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        line-height: 1.5;
        border: 1px solid;
    }
    .status-good {
        background: rgba(16,185,129,0.10);
        border-color: rgba(52,211,153,0.28);
        color: #d1fae5;
    }
    .status-watch {
        background: rgba(245,158,11,0.10);
        border-color: rgba(251,191,36,0.30);
        color: #fef3c7;
    }
    .status-alert {
        background: rgba(239,68,68,0.10);
        border-color: rgba(248,113,113,0.30);
        color: #fee2e2;
    }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(18, 43, 68, 0.96),
            rgba(7, 25, 43, 0.96)
        );
        border: 1px solid rgba(103,232,249,0.13);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* ---------- Upload box ---------- */
    [data-testid="stFileUploader"] {
        background: rgba(18, 54, 84, 0.92);
        border: 1px dashed rgba(34,211,238,0.48);
        border-radius: 14px;
        padding: 16px;
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(148,163,184,0.14);
    }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] {
        color: #94a3b8;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fafc;
    }

    /* ---------- Dataframe ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(103,232,249,0.10);
    }

    /* ---------- Download button ---------- */
    .stDownloadButton button {
        width: 100%;
        min-height: 46px;
        border-radius: 10px;
        border: 1px solid rgba(34,211,238,0.52);
        background: linear-gradient(90deg, #0369a1, #0891b2);
        color: white;
        font-weight: 750;
    }

    .stDownloadButton button:hover {
        border-color: #67e8f9;
        color: white;
    }

    /* ---------- Footer ---------- */
    .footer-note {
        margin-top: 22px;
        padding: 18px;
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.7;
        border-top: 1px solid rgba(148,163,184,0.12);
    }


    /* ---------- Hero / illustrative visuals ---------- */
    .hero-panel {
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: 1.45fr 1fr;
        align-items: center;
        gap: 24px;
        min-height: 225px;
        padding: 30px 34px;
        margin-bottom: 22px;
        border-radius: 20px;
        background:
            linear-gradient(120deg, rgba(28, 79, 118, 0.97), rgba(18, 58, 91, 0.95));
        border: 1px solid rgba(103, 232, 249, 0.28);
        box-shadow:
            0 16px 42px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.06);
    }

    .hero-panel:before {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        right: -120px;
        top: -150px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(34,211,238,0.24), rgba(34,211,238,0));
    }

    .hero-kicker {
        color: #67e8f9;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(28px, 3vw, 42px);
        font-weight: 850;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    .hero-copy {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.7;
        max-width: 720px;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }

    .hero-tag {
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(103,232,249,0.10);
        border: 1px solid rgba(103,232,249,0.22);
        color: #dffaff;
        font-size: 12px;
        font-weight: 700;
    }

    .hero-art {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 180px;
        filter: drop-shadow(0 0 18px rgba(34,211,238,0.20));
    }

    .visual-card {
        padding: 16px;
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(37,82,119,0.82), rgba(17,51,80,0.90));
        border: 1px solid rgba(125,211,252,0.24);
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        margin-bottom: 14px;
    }

    .mini-visual-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 8px 0 20px;
    }

    .mini-visual {
        min-height: 100px;
        border-radius: 14px;
        padding: 14px;
        background: linear-gradient(145deg, rgba(34,77,112,0.92), rgba(20,58,91,0.94));
        border: 1px solid rgba(103,232,249,0.18);
        text-align: center;
    }

    .mini-visual .icon {
        font-size: 31px;
        margin-bottom: 6px;
    }

    .mini-visual .label {
        color: #dbeafe;
        font-weight: 750;
        font-size: 13px;
    }

    .mini-visual .caption {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 3px;
    }

    /* Brighter upload and tabs */
    [data-testid="stFileUploader"] section {
        background: rgba(34, 76, 112, 0.56);
        border-radius: 12px;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(25, 65, 99, 0.55);
        padding: 6px;
        border-radius: 12px;
    }

    /* ---------- Responsive ---------- */
    @media (max-width: 900px) {
        .nav-items { display: none; }
        .brand { font-size: 19px; }
        .top-nav { padding: 13px 16px; }
        .score-number { font-size: 44px; }
        .hero-panel { grid-template-columns: 1fr; padding: 24px; }
        .hero-art { min-height: 140px; }
        .mini-visual-row { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. Header / Navigation
# =========================================================

st.markdown(
    """
    <div class="top-nav">
        <div class="brand">
            <span class="brand-icon">▣</span>
            Medical Gait AI
        </div>

        <div class="nav-items">
            <span class="nav-active">Dashboard</span>
            <span>Patient Data</span>
            <span>Reports</span>
            <span>Settings</span>
        </div>

        <div class="nav-user">👤</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-panel">
        <div>
            <div class="hero-kicker">AI-ASSISTED MOVEMENT ANALYSIS</div>
            <div class="hero-title">Video Gait Analysis Dashboard</div>
            <div class="hero-copy">
                วิเคราะห์การเคลื่อนไหวจากวิดีโอด้วย MediaPipe Pose
                พร้อมสรุปมุมข้อต่อ ความสมมาตร ROM และ Gait Screening
                ในรูปแบบแดชบอร์ดที่อ่านผลได้ง่าย
            </div>
            <div class="hero-tags">
                <span class="hero-tag">Pose Tracking</span>
                <span class="hero-tag">Hip / Knee / Ankle</span>
                <span class="hero-tag">Symmetry Index</span>
                <span class="hero-tag">ROM Analysis</span>
            </div>
        </div>

        <div class="hero-art" aria-label="ภาพประกอบระบบติดตามท่าเดิน">
            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgwIiBoZWlnaHQ9IjE5MCIgdmlld0JveD0iMCAwIDI4MCAxOTAiCiAgICAgICAgICAgICAgICAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiByb2xlPSJpbWciPgogICAgICAgICAgICAgICAgPGRlZnM+CiAgICAgICAgICAgICAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJsaW1iIiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+CiAgICAgICAgICAgICAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNlMGYyZmUiLz4KICAgICAgICAgICAgICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjOTNjNWZkIi8+CiAgICAgICAgICAgICAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgICAgICAgICAgICAgICA8ZmlsdGVyIGlkPSJnbG93Ij4KICAgICAgICAgICAgICAgICAgICAgICAgPGZlR2F1c3NpYW5CbHVyIHN0ZERldmlhdGlvbj0iMy4yIiByZXN1bHQ9ImJsdXIiLz4KICAgICAgICAgICAgICAgICAgICAgICAgPGZlTWVyZ2U+CiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8ZmVNZXJnZU5vZGUgaW49ImJsdXIiLz4KICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxmZU1lcmdlTm9kZSBpbj0iU291cmNlR3JhcGhpYyIvPgogICAgICAgICAgICAgICAgICAgICAgICA8L2ZlTWVyZ2U+CiAgICAgICAgICAgICAgICAgICAgPC9maWx0ZXI+CiAgICAgICAgICAgICAgICA8L2RlZnM+CgogICAgICAgICAgICAgICAgPGNpcmNsZSBjeD0iMTM0IiBjeT0iMjgiIHI9IjE2IiBmaWxsPSIjZWFmOGZmIi8+CiAgICAgICAgICAgICAgICA8ZyBzdHJva2U9InVybCgjbGltYikiIHN0cm9rZS13aWR0aD0iNyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIj4KICAgICAgICAgICAgICAgICAgICA8bGluZSB4MT0iMTM0IiB5MT0iNDciIHgyPSIxMzEiIHkyPSI4OCIvPgogICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIxMzEiIHkxPSI2MiIgeDI9Ijk2IiB5Mj0iNzYiLz4KICAgICAgICAgICAgICAgICAgICA8bGluZSB4MT0iOTYiIHkxPSI3NiIgeDI9Ijc2IiB5Mj0iMTA4Ii8+CiAgICAgICAgICAgICAgICAgICAgPGxpbmUgeDE9IjEzMSIgeTE9IjYyIiB4Mj0iMTY2IiB5Mj0iNzgiLz4KICAgICAgICAgICAgICAgICAgICA8bGluZSB4MT0iMTY2IiB5MT0iNzgiIHgyPSIyMDAiIHkyPSI5MiIvPgogICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIxMzEiIHkxPSI4OCIgeDI9IjEwNSIgeTI9IjEyNSIvPgogICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIxMDUiIHkxPSIxMjUiIHgyPSI3MiIgeTI9IjE2MCIvPgogICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIxMzEiIHkxPSI4OCIgeDI9IjE2MCIgeTI9IjEyNSIvPgogICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIxNjAiIHkxPSIxMjUiIHgyPSIxODAiIHkyPSIxNjQiLz4KICAgICAgICAgICAgICAgIDwvZz4KCiAgICAgICAgICAgICAgICA8ZyBmaWx0ZXI9InVybCgjZ2xvdykiPgogICAgICAgICAgICAgICAgICAgIDxjaXJjbGUgY3g9IjEzMSIgY3k9IjYyIiByPSI4IiBmaWxsPSIjMjJkM2VlIi8+CiAgICAgICAgICAgICAgICAgICAgPGNpcmNsZSBjeD0iOTYiIGN5PSI3NiIgcj0iNyIgZmlsbD0iIzY3ZThmOSIvPgogICAgICAgICAgICAgICAgICAgIDxjaXJjbGUgY3g9Ijc2IiBjeT0iMTA4IiByPSI3IiBmaWxsPSIjMjJkM2VlIi8+CiAgICAgICAgICAgICAgICAgICAgPGNpcmNsZSBjeD0iMTY2IiBjeT0iNzgiIHI9IjciIGZpbGw9IiNjMDg0ZmMiLz4KICAgICAgICAgICAgICAgICAgICA8Y2lyY2xlIGN4PSIyMDAiIGN5PSI5MiIgcj0iNyIgZmlsbD0iI2E4NTVmNyIvPgogICAgICAgICAgICAgICAgICAgIDxjaXJjbGUgY3g9IjEzMSIgY3k9Ijg4IiByPSI5IiBmaWxsPSIjNjdlOGY5Ii8+CiAgICAgICAgICAgICAgICAgICAgPGNpcmNsZSBjeD0iMTA1IiBjeT0iMTI1IiByPSI4IiBmaWxsPSIjMjJkM2VlIi8+CiAgICAgICAgICAgICAgICAgICAgPGNpcmNsZSBjeD0iNzIiIGN5PSIxNjAiIHI9IjgiIGZpbGw9IiM2N2U4ZjkiLz4KICAgICAgICAgICAgICAgICAgICA8Y2lyY2xlIGN4PSIxNjAiIGN5PSIxMjUiIHI9IjgiIGZpbGw9IiNjMDg0ZmMiLz4KICAgICAgICAgICAgICAgICAgICA8Y2lyY2xlIGN4PSIxODAiIGN5PSIxNjQiIHI9IjgiIGZpbGw9IiNhODU1ZjciLz4KICAgICAgICAgICAgICAgIDwvZz4KCiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTggMTQ1IEM1NSAxMjUsIDYzIDE1OCwgOTggMTQyIFMxNTAgMTE4LCAxOTQgMTM2IFMyMzggMTQ4LCAyNjYgMTE4IgogICAgICAgICAgICAgICAgICAgICAgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjJkM2VlIiBzdHJva2Utd2lkdGg9IjIuNSIgb3BhY2l0eT0iMC43NSIvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTE4IDE1NSBIMjY1IiBzdHJva2U9IiM3ZGQzZmMiIHN0cm9rZS13aWR0aD0iMSIgb3BhY2l0eT0iMC4xOCIvPgogICAgICAgICAgICAgICAgPGNpcmNsZSBjeD0iMjMyIiBjeT0iNDIiIHI9IjIyIiBmaWxsPSJub25lIiBzdHJva2U9IiM2N2U4ZjkiIHN0cm9rZS13aWR0aD0iMiIgb3BhY2l0eT0iMC41NSIvPgogICAgICAgICAgICAgICAgPGNpcmNsZSBjeD0iMjMyIiBjeT0iNDIiIHI9IjkiIGZpbGw9IiMyMmQzZWUiIG9wYWNpdHk9IjAuNjUiLz4KICAgICAgICAgICAgPC9zdmc+" alt="ภาพประกอบระบบติดตามท่าเดิน" style="width:280px;max-width:100%;height:auto;display:block;margin:auto;" />
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="mini-visual-row">
        <div class="mini-visual">
            <div class="icon">🧍</div>
            <div class="label">Pose Detection</div>
            <div class="caption">ตรวจจับโครงร่างจากวิดีโอ</div>
        </div>
        <div class="mini-visual">
            <div class="icon">📐</div>
            <div class="label">Joint Angles</div>
            <div class="caption">Hip · Knee · Ankle</div>
        </div>
        <div class="mini-visual">
            <div class="icon">⚖️</div>
            <div class="label">Symmetry</div>
            <div class="caption">เปรียบเทียบซ้าย–ขวา</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 5. ฟังก์ชันคำนวณมุมและแปลงเป็นมุมเชิงคลินิก
# =========================================================

def calculate_angle(a, b, c):
    """
    คำนวณมุม ABC แบบเรขาคณิต 2D (0-180 องศา)
    จากจุด A-B-C โดย B เป็นจุดยอดมุม
    """

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


def clinical_knee_flexion(hip, knee, ankle):
    """
    MediaPipe geometric knee angle:
        fully extended ~ 180°
    Clinical knee flexion:
        fully extended ~ 0°

    ดังนั้น:
        Knee Flexion = 180 - geometric angle
    """

    raw_angle = calculate_angle(
        hip,
        knee,
        ankle
    )

    return float(
        180.0 - raw_angle
    )


def clinical_ankle_dorsiflexion(knee, ankle, foot):
    """
    ในภาพด้านข้าง:
        tibia-foot geometric angle ~ 90° ที่ neutral

    กำหนด:
        dorsiflexion  = ค่าบวก
        plantarflexion = ค่าลบ

    ดังนั้น:
        Ankle DF = 90 - geometric angle
    """

    raw_angle = calculate_angle(
        knee,
        ankle,
        foot
    )

    return float(
        90.0 - raw_angle
    )


def clinical_hip_flexion(
    shoulder,
    hip,
    knee,
    direction_sign
):
    """
    แปลง geometric hip angle ให้เป็นมุมเชิงคลินิกอย่างง่าย
    สำหรับวิดีโอด้านข้าง (sagittal view)

    flexion  = ค่าบวก
    extension = ค่าลบ

    magnitude = 180 - geometric angle

    เครื่องหมายถูกกำหนดจากตำแหน่งเข่าเทียบสะโพก
    ตามทิศทางการเดินที่ผู้ใช้เลือก
    """

    raw_angle = calculate_angle(
        shoulder,
        hip,
        knee
    )

    magnitude = max(
        0.0,
        180.0 - raw_angle
    )

    forward_position = (
        knee[0] - hip[0]
    ) * direction_sign

    if forward_position >= 0:
        return float(magnitude)

    return float(-magnitude)


# =========================================================
# 6. Symmetry Index / ROM
# =========================================================

def calculate_symmetry_index(left_val, right_val):
    """
    Symmetry Index (%)

        |L - R|
    ---------------- x 100
     0.5(|L|+|R|)

    ใช้กับค่าที่เป็น scalar เช่น ROM หรือ peak
    """

    denominator = 0.5 * (
        abs(left_val) + abs(right_val)
    )

    if denominator < 1e-9:
        return 0.0

    return float(
        (
            abs(left_val - right_val)
            / denominator
        ) * 100.0
    )


def calculate_rom(series):
    """
    Range of Motion = maximum - minimum
    """

    if len(series) == 0:
        return 0.0

    return float(
        np.nanmax(series)
        - np.nanmin(series)
    )


def calculate_curve_mae(
    left_curve,
    right_curve
):
    """
    Mean Absolute Error (MAE) ของเส้นโค้งซ้าย-ขวา หน่วยเป็นองศา

    เหตุผลที่ใช้ MAE:
    - ไม่หารด้วยค่ามุมเฉลี่ย
    - ไม่เกิดค่า SI พุ่งสูงเมื่อกราฟผ่าน 0°
    - ตีความตรงไปตรงมา เช่น MAE = 4.2° หมายถึง
      มุมซ้าย-ขวาต่างกันเฉลี่ย 4.2° ตลอด gait cycle
    """

    left_curve = np.asarray(
        left_curve,
        dtype=float
    )

    right_curve = np.asarray(
        right_curve,
        dtype=float
    )

    valid = (
        np.isfinite(left_curve)
        & np.isfinite(right_curve)
    )

    if valid.sum() < 5:
        return np.nan

    return float(
        np.mean(
            np.abs(
                left_curve[valid]
                - right_curve[valid]
            )
        )
    )


def mae_to_score(
    mae_deg,
    tolerance_deg
):
    """
    แปลง MAE (องศา) เป็นคะแนน 0-100

    tolerance_deg เป็น system scaling constant
    ไม่ใช่เกณฑ์วินิจฉัยทางคลินิก
    """

    if not np.isfinite(mae_deg):
        return np.nan

    if tolerance_deg <= 0:
        return 0.0

    return float(
        np.clip(
            100.0 * (
                1.0
                - mae_deg / tolerance_deg
            ),
            0.0,
            100.0
        )
    )


def rom_si_to_score(
    rom_si,
    tolerance_percent=20.0
):
    """
    แปลง ROM Symmetry Index (%) เป็นคะแนน 0-100

    0%  = ROM ซ้าย-ขวาเท่ากัน
    tolerance_percent = system scaling constant
    """

    if not np.isfinite(rom_si):
        return np.nan

    return float(
        np.clip(
            100.0 * (
                1.0
                - rom_si / tolerance_percent
            ),
            0.0,
            100.0
        )
    )


# System scaling constants สำหรับคะแนน prototype
# ไม่ใช่ clinical cut-off
CURVE_MAE_TOLERANCE_DEG = {
    "Hip": 15.0,
    "Knee": 15.0,
    "Ankle": 12.0,
}

PHASE_MAE_TOLERANCE_DEG = 12.0
ROM_SI_TOLERANCE_PERCENT = 20.0


# =========================================================
# 7. Gait-cycle detection
# =========================================================

GAIT_PHASES = [
    ("Initial Contact", 0, 2),
    ("Loading Response", 0, 10),
    ("Mid Stance", 10, 30),
    ("Terminal Stance", 30, 50),
    ("Pre-Swing", 50, 60),
    ("Initial Swing", 60, 73),
    ("Mid Swing", 73, 87),
    ("Terminal Swing", 87, 100),
]


def smooth_series(values, window=5):
    """
    Moving average แบบง่าย โดยไม่ต้องใช้ scipy
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 3:
        return values.copy()

    window = int(
        max(
            3,
            min(
                window,
                len(values)
            )
        )
    )

    if window % 2 == 0:
        window += 1

    pad = window // 2

    padded = np.pad(
        values,
        (pad, pad),
        mode="edge"
    )

    kernel = np.ones(window) / window

    return np.convolve(
        padded,
        kernel,
        mode="valid"
    )


def detect_heel_strikes(
    forward_heel_position,
    fps,
    min_interval_s=0.55,
    prominence=0.015
):
    """
    ตรวจ heel-strike แบบ heuristic จากตำแหน่งส้นเท้า
    ที่อยู่ด้านหน้าสุดเมื่อเทียบกับ pelvis

    เหมาะกับ:
    - วิดีโอด้านข้าง
    - ผู้เดินผ่านกล้องหรือ treadmill ที่เห็นเท้าชัด

    ไม่ใช่ force-plate event detection
    """

    values = np.asarray(
        forward_heel_position,
        dtype=float
    )

    if len(values) < 10:
        return []

    smooth_window = max(
        3,
        int(
            round(
                fps * 0.08
            )
        )
    )

    smoothed = smooth_series(
        values,
        smooth_window
    )

    min_distance = max(
        3,
        int(
            round(
                fps * min_interval_s
            )
        )
    )

    candidates = []

    for i in range(
        1,
        len(smoothed) - 1
    ):

        if not (
            smoothed[i] >= smoothed[i - 1]
            and smoothed[i] > smoothed[i + 1]
        ):
            continue

        left = max(
            0,
            i - min_distance // 2
        )

        right = min(
            len(smoothed),
            i + min_distance // 2 + 1
        )

        local_min = np.nanmin(
            smoothed[left:right]
        )

        if (
            smoothed[i] - local_min
        ) >= prominence:

            candidates.append(i)

    if not candidates:
        return []

    selected = []

    for idx in candidates:

        if not selected:

            selected.append(idx)
            continue

        if (
            idx - selected[-1]
        ) >= min_distance:

            selected.append(idx)

        elif (
            smoothed[idx]
            > smoothed[selected[-1]]
        ):

            selected[-1] = idx

    return selected


def normalize_cycles(
    df,
    strike_indices,
    joint_columns
):
    """
    Normalize รอบการเดินจาก heel strike หนึ่ง
    ไป heel strike ครั้งถัดไปของเท้าข้างเดียว
    เป็น 0-100% ด้วย interpolation 101 จุด
    """

    if len(strike_indices) < 2:
        return None

    percent = np.linspace(
        0,
        100,
        101
    )

    normalized = {
        column: []
        for column in joint_columns
    }

    cycle_durations = []

    valid_cycle_count = 0

    for start, end in zip(
        strike_indices[:-1],
        strike_indices[1:]
    ):

        if end - start < 8:
            continue

        segment = df.iloc[
            start:end + 1
        ]

        segment_time = (
            segment["Time (s)"].to_numpy(
                dtype=float
            )
        )

        duration = (
            segment_time[-1]
            - segment_time[0]
        )

        if duration <= 0:
            continue

        x_old = np.linspace(
            0,
            100,
            len(segment)
        )

        cycle_ok = True

        temp = {}

        for column in joint_columns:

            y = segment[
                column
            ].to_numpy(
                dtype=float
            )

            valid = np.isfinite(y)

            if valid.sum() < 5:
                cycle_ok = False
                break

            temp[column] = np.interp(
                percent,
                x_old[valid],
                y[valid]
            )

        if not cycle_ok:
            continue

        for column in joint_columns:
            normalized[column].append(
                temp[column]
            )

        cycle_durations.append(
            duration
        )

        valid_cycle_count += 1

    if valid_cycle_count == 0:
        return None

    mean_curves = {}

    for column in joint_columns:

        mean_curves[column] = np.mean(
            np.vstack(
                normalized[column]
            ),
            axis=0
        )

    return {
        "percent": percent,
        "mean_curves": mean_curves,
        "cycle_count": valid_cycle_count,
        "cycle_durations": cycle_durations,
        "mean_cycle_duration": float(
            np.mean(
                cycle_durations
            )
        )
    }


def build_gait_cycle_analysis(
    df,
    fps
):
    """
    สร้าง gait cycle แยกซ้าย/ขวา
    แล้ว normalize เป็น 0-100%
    """

    required = [
        "Left Heel Forward",
        "Right Heel Forward",
        "Left Hip Flexion",
        "Right Hip Flexion",
        "Left Knee Flexion",
        "Right Knee Flexion",
        "Left Ankle DF",
        "Right Ankle DF",
    ]

    if any(
        column not in df.columns
        for column in required
    ):
        return None

    left_strikes = detect_heel_strikes(
        df[
            "Left Heel Forward"
        ].to_numpy(),
        fps
    )

    right_strikes = detect_heel_strikes(
        df[
            "Right Heel Forward"
        ].to_numpy(),
        fps
    )

    left_cycle = normalize_cycles(
        df,
        left_strikes,
        [
            "Left Hip Flexion",
            "Left Knee Flexion",
            "Left Ankle DF",
        ]
    )

    right_cycle = normalize_cycles(
        df,
        right_strikes,
        [
            "Right Hip Flexion",
            "Right Knee Flexion",
            "Right Ankle DF",
        ]
    )

    if (
        left_cycle is None
        or right_cycle is None
    ):
        return {
            "available": False,
            "left_strikes": left_strikes,
            "right_strikes": right_strikes,
            "left_cycle": left_cycle,
            "right_cycle": right_cycle,
        }

    return {
        "available": True,
        "left_strikes": left_strikes,
        "right_strikes": right_strikes,
        "left_cycle": left_cycle,
        "right_cycle": right_cycle,
    }


# =========================================================
# 8. Phase metrics / reference bands
# =========================================================

def curve_value_at(
    percent,
    curve,
    target_percent
):

    return float(
        np.interp(
            target_percent,
            percent,
            curve
        )
    )


def curve_range_metric(
    percent,
    curve,
    start_percent,
    end_percent,
    mode="mean"
):

    percent = np.asarray(
        percent
    )

    curve = np.asarray(
        curve
    )

    mask = (
        (percent >= start_percent)
        & (percent <= end_percent)
    )

    values = curve[mask]

    if len(values) == 0:
        return np.nan

    if mode == "max":
        return float(
            np.nanmax(values)
        )

    if mode == "min":
        return float(
            np.nanmin(values)
        )

    return float(
        np.nanmean(values)
    )


def extract_phase_metrics(
    percent,
    hip_curve,
    knee_curve,
    ankle_curve
):
    """
    Extract metrics จาก joint-angle curve
    ใน sagittal plane
    """

    return {
        "Hip at Initial Contact": curve_value_at(
            percent,
            hip_curve,
            0
        ),
        "Max Hip Extension (30-60%)": curve_range_metric(
            percent,
            hip_curve,
            30,
            60,
            "min"
        ),
        "Max Hip Flexion in Swing": curve_range_metric(
            percent,
            hip_curve,
            60,
            100,
            "max"
        ),

        "Knee at Initial Contact": curve_value_at(
            percent,
            knee_curve,
            0
        ),
        "Peak Knee Flexion Loading": curve_range_metric(
            percent,
            knee_curve,
            0,
            10,
            "max"
        ),
        "Knee Mid-Stance": curve_range_metric(
            percent,
            knee_curve,
            10,
            30,
            "mean"
        ),
        "Peak Knee Flexion Swing": curve_range_metric(
            percent,
            knee_curve,
            60,
            87,
            "max"
        ),

        "Ankle at Initial Contact": curve_value_at(
            percent,
            ankle_curve,
            0
        ),
        "Ankle Mid-Stance DF": curve_range_metric(
            percent,
            ankle_curve,
            10,
            30,
            "mean"
        ),
        "Ankle Terminal-Stance DF": curve_range_metric(
            percent,
            ankle_curve,
            30,
            50,
            "max"
        ),
        "Ankle Mid-Swing DF": curve_range_metric(
            percent,
            ankle_curve,
            73,
            87,
            "mean"
        ),
    }


REFERENCE_BANDS = {
    # Hip/Knee bands are broad educational sagittal-plane references.
    # Ankle bands use the Thai healthy-adult study as a contextual reference.
    "Hip at Initial Contact": (20.0, 30.0),
    "Max Hip Extension (30-60%)": (-20.0, -10.0),
    "Max Hip Flexion in Swing": (25.0, 35.0),

    "Knee at Initial Contact": (0.0, 10.0),
    "Peak Knee Flexion Loading": (10.0, 20.0),
    "Knee Mid-Stance": (0.0, 10.0),
    "Peak Knee Flexion Swing": (50.0, 70.0),

    "Ankle at Initial Contact": (-5.0, 5.0),
    "Ankle Mid-Stance DF": (4.0, 10.0),
    "Ankle Terminal-Stance DF": (9.0, 18.0),
    "Ankle Mid-Swing DF": (0.0, 7.0),
}


def reference_status(
    value,
    metric_name
):

    if metric_name not in REFERENCE_BANDS:
        return "—"

    low, high = REFERENCE_BANDS[
        metric_name
    ]

    if low <= value <= high:
        return "อยู่ในช่วงอ้างอิง"

    return "นอกช่วงอ้างอิง"


def calculate_gait_screening(
    df,
    cycle_analysis=None
):
    """
    Gait Screening Score แบบ prototype

    ถ้ามี normalized gait cycle:
        40% Joint Curve Similarity
        35% ROM Symmetry
        25% Peak/Phase Similarity

    ถ้ายังตรวจ gait cycle ไม่พอ:
        ใช้ ROM Symmetry เป็น fallback
        และระบุ confidence ว่าจำกัด

    หมายเหตุ:
    - Curve ใช้ MAE หน่วยองศา ไม่ใช้ Symmetry Index แบบหารด้วยมุมเฉลี่ย
    - ROM ยังคงใช้ Symmetry Index ได้ เพราะ ROM เป็น scalar บวก
    - คะแนนนี้เป็น system screening score ไม่ใช่ clinical validated score
    """

    joint_pairs = [
        (
            "Hip",
            "Left Hip Flexion",
            "Right Hip Flexion",
        ),
        (
            "Knee",
            "Left Knee Flexion",
            "Right Knee Flexion",
        ),
        (
            "Ankle",
            "Left Ankle DF",
            "Right Ankle DF",
        ),
    ]

    # =====================================================
    # 1) ROM symmetry
    # =====================================================

    rom_si = {}
    rom_scores = {}

    for (
        joint,
        left_col,
        right_col
    ) in joint_pairs:

        left_rom = calculate_rom(
            df[left_col]
        )

        right_rom = calculate_rom(
            df[right_col]
        )

        rom_si[joint] = (
            calculate_symmetry_index(
                left_rom,
                right_rom
            )
        )

        rom_scores[joint] = (
            rom_si_to_score(
                rom_si[joint],
                ROM_SI_TOLERANCE_PERCENT
            )
        )

    overall_rom_si = float(
        np.mean(
            list(
                rom_si.values()
            )
        )
    )

    rom_component_score = float(
        np.mean(
            list(
                rom_scores.values()
            )
        )
    )

    # =====================================================
    # 2) Curve similarity + phase similarity
    # =====================================================

    curve_mae = {
        "Hip": np.nan,
        "Knee": np.nan,
        "Ankle": np.nan,
    }

    curve_scores = {
        "Hip": np.nan,
        "Knee": np.nan,
        "Ankle": np.nan,
    }

    phase_mae = np.nan
    phase_component_score = np.nan
    curve_component_score = np.nan

    cycle_available = (
        cycle_analysis is not None
        and cycle_analysis.get(
            "available",
            False
        )
    )

    if cycle_available:

        left_cycle = cycle_analysis[
            "left_cycle"
        ]

        right_cycle = cycle_analysis[
            "right_cycle"
        ]

        left = left_cycle[
            "mean_curves"
        ]

        right = right_cycle[
            "mean_curves"
        ]

        curve_mae["Hip"] = (
            calculate_curve_mae(
                left[
                    "Left Hip Flexion"
                ],
                right[
                    "Right Hip Flexion"
                ]
            )
        )

        curve_mae["Knee"] = (
            calculate_curve_mae(
                left[
                    "Left Knee Flexion"
                ],
                right[
                    "Right Knee Flexion"
                ]
            )
        )

        curve_mae["Ankle"] = (
            calculate_curve_mae(
                left[
                    "Left Ankle DF"
                ],
                right[
                    "Right Ankle DF"
                ]
            )
        )

        for joint in [
            "Hip",
            "Knee",
            "Ankle"
        ]:

            curve_scores[joint] = (
                mae_to_score(
                    curve_mae[joint],
                    CURVE_MAE_TOLERANCE_DEG[
                        joint
                    ]
                )
            )

        finite_curve_scores = [
            value
            for value in curve_scores.values()
            if np.isfinite(value)
        ]

        curve_component_score = float(
            np.mean(
                finite_curve_scores
            )
        )

        # ---------------------------------------------
        # Phase / peak similarity
        # ---------------------------------------------

        gait_percent = left_cycle[
            "percent"
        ]

        left_metrics = extract_phase_metrics(
            gait_percent,
            left[
                "Left Hip Flexion"
            ],
            left[
                "Left Knee Flexion"
            ],
            left[
                "Left Ankle DF"
            ]
        )

        right_metrics = extract_phase_metrics(
            gait_percent,
            right[
                "Right Hip Flexion"
            ],
            right[
                "Right Knee Flexion"
            ],
            right[
                "Right Ankle DF"
            ]
        )

        phase_differences = []

        for metric_name in left_metrics:

            left_value = left_metrics[
                metric_name
            ]

            right_value = right_metrics[
                metric_name
            ]

            if (
                np.isfinite(left_value)
                and np.isfinite(right_value)
            ):

                phase_differences.append(
                    abs(
                        left_value
                        - right_value
                    )
                )

        if phase_differences:

            phase_mae = float(
                np.mean(
                    phase_differences
                )
            )

            phase_component_score = (
                mae_to_score(
                    phase_mae,
                    PHASE_MAE_TOLERANCE_DEG
                )
            )

        else:

            phase_component_score = 0.0

        # ---------------------------------------------
        # Final weighted score
        # ---------------------------------------------

        score = float(
            np.clip(
                0.40
                * curve_component_score
                + 0.35
                * rom_component_score
                + 0.25
                * phase_component_score,
                0.0,
                100.0
            )
        )

        overall_curve_mae = float(
            np.mean(
                [
                    value
                    for value
                    in curve_mae.values()
                    if np.isfinite(value)
                ]
            )
        )

        confidence = (
            "สูงขึ้น "
            "(ตรวจพบ normalized gait cycle)"
        )

    else:

        # ไม่มี gait cycle:
        # ไม่สร้าง curve score ปลอม
        overall_curve_mae = np.nan

        score = rom_component_score

        confidence = (
            "จำกัด "
            "(ยังตรวจ gait cycle ได้ไม่เพียงพอ; "
            "คะแนนอาศัย ROM symmetry เป็นหลัก)"
        )

    # =====================================================
    # 3) Status จาก final score
    # =====================================================
    # เป็น system threshold สำหรับ prototype เท่านั้น

    if score >= 80:

        level = "normal"

        status = (
            "🟢 ความสมมาตรโดยรวมอยู่ในระดับดี"
        )

        description = (
            "คะแนนระบบอยู่ในระดับสูงจากการเปรียบเทียบ "
            "joint-angle curves, ROM และ phase metrics "
            "ที่ระบบตรวจได้"
        )

    elif score >= 60:

        level = "warning"

        status = (
            "🟡 พบความแตกต่างบางส่วน"
        )

        description = (
            "ระบบพบความแตกต่างซ้าย–ขวาบางส่วน "
            "ควรพิจารณากราฟ gait cycle, ROM "
            "และ phase metrics ร่วมกัน"
        )

    else:

        level = "danger"

        status = (
            "🔴 พบความแตกต่างซ้าย–ขวาค่อนข้างมาก"
        )

        description = (
            "ระบบพบความแตกต่างซ้าย–ขวาหลายองค์ประกอบ "
            "ควรตรวจคุณภาพวิดีโอและพิจารณาการประเมินเพิ่มเติม"
        )

    recommendation = (
        "ใช้คะแนนนี้เป็นข้อมูลคัดกรองเชิงระบบเท่านั้น "
        "ควรอ่านร่วมกับค่า Curve MAE (°), ROM Symmetry Index (%), "
        "phase/peak difference และกราฟ 0–100% gait cycle "
        "ไม่ควรใช้คะแนนเดียวเป็นการวินิจฉัย"
    )

    return {
        "status": status,
        "level": level,
        "score": score,

        "overall_curve_mae": overall_curve_mae,
        "overall_rom_si": overall_rom_si,
        "phase_mae": phase_mae,

        "curve_component_score":
            curve_component_score,
        "rom_component_score":
            rom_component_score,
        "phase_component_score":
            phase_component_score,

        "hip_curve_mae":
            curve_mae["Hip"],
        "knee_curve_mae":
            curve_mae["Knee"],
        "ankle_curve_mae":
            curve_mae["Ankle"],

        "hip_curve_score":
            curve_scores["Hip"],
        "knee_curve_score":
            curve_scores["Knee"],
        "ankle_curve_score":
            curve_scores["Ankle"],

        "hip_rom_si": rom_si["Hip"],
        "knee_rom_si": rom_si["Knee"],
        "ankle_rom_si": rom_si["Ankle"],

        "description": description,
        "recommendation": recommendation,
        "confidence": confidence,
        "cycle_available": cycle_available,
    }


# =========================================================
# 9. Sidebar - ข้อมูลประกอบการวิเคราะห์
# =========================================================

with st.sidebar:

    st.markdown("## 🦶 Gait Analysis")

    st.markdown(
        """
        **รูปแบบวิดีโอที่แนะนำ**

        - กล้องด้านข้าง (sagittal view)
        - เห็นศีรษะถึงเท้าหรืออย่างน้อยลำตัวถึงเท้า
        - เห็นส้นเท้าและปลายเท้าชัด
        - กล้องนิ่ง
        - เดินด้วยความเร็วธรรมชาติ
        """
    )

    st.divider()

    st.markdown(
        "### 👤 ข้อมูลประกอบรายงาน"
    )

    subject_age = st.number_input(
        "อายุ (ปี)",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

    subject_sex = st.selectbox(
        "เพศที่บันทึกในรายงาน",
        [
            "ไม่ระบุ",
            "ชาย",
            "หญิง",
            "อื่น ๆ"
        ]
    )

    st.caption(
        "อายุและเพศถูกใช้เป็นข้อมูลประกอบรายงานเท่านั้น "
        "ไม่ได้ถูกใช้ตัดสินว่าปกติหรือผิดปกติในคะแนนระบบ"
    )

    st.divider()

    walking_direction = st.radio(
        "ทิศทางการเดินในภาพ",
        [
            "เดินไปทางขวา →",
            "← เดินไปทางซ้าย"
        ],
        help=(
            "ใช้เพื่อกำหนดเครื่องหมาย hip flexion/extension "
            "และตำแหน่งส้นเท้าด้านหน้า"
        )
    )

    direction_sign = (
        1
        if walking_direction
        == "เดินไปทางขวา →"
        else -1
    )

    st.divider()

    st.markdown("### ⚠️ ข้อควรทราบ")

    st.caption(
        "ระบบนี้เป็น markerless 2D video analysis "
        "จึงไม่เทียบเท่าห้อง gait laboratory ที่ใช้ "
        "3D motion capture และ force plate"
    )


# =========================================================
# 10. อัปโหลดวิดีโอ
# =========================================================

st.markdown(
    '<div class="section-title">📹 อัปโหลดวิดีโอด้านข้าง</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'ระบบจะพยายามตรวจ heel-strike และ normalize การเดินเป็น 0–100% gait cycle '
    'หากข้อมูลเพียงพอ'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "เลือกคลิปวิดีโอการเดิน",
    type=[
        "mp4",
        "mov",
        "avi"
    ],
    help="แนะนำให้ใช้วิดีโอที่เห็นร่างกายเต็มตัวและมีแสงเพียงพอ"
)


# =========================================================
# 11. เริ่มวิเคราะห์
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # บันทึกไฟล์ชั่วคราว
    # =====================================================

    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    tfile.write(
        uploaded_file.read()
    )

    tfile.close()

    video_path = tfile.name


    # =====================================================
    # เปิดวิดีโอ
    # =====================================================

    cap = cv2.VideoCapture(
        video_path
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    ) or 1

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0


    duration = total_frames / fps


    # =====================================================
    # ข้อมูลวิดีโอ
    # =====================================================

    info1, info2, info3 = st.columns(3)

    with info1:

        st.metric(
            "🎞️ จำนวนเฟรม",
            f"{total_frames:,}"
        )

    with info2:

        st.metric(
            "⏱️ FPS",
            f"{fps:.1f}"
        )

    with info3:

        st.metric(
            "🕐 ความยาว",
            f"{duration:.1f} วินาที"
        )


    # =====================================================
    # เตรียมตัวแปร
    # =====================================================

    frames_data = []

    last_pose_image = None

    frame_count = 0

    st_frame = st.empty()

    progress_bar = st.progress(0)

    status_text = st.empty()


    # =====================================================
    # MediaPipe Pose
    # =====================================================

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1


            # -------------------------------------------------
            # BGR → RGB
            # -------------------------------------------------

            image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image.flags.writeable = False


            # -------------------------------------------------
            # Pose
            # -------------------------------------------------

            results = pose.process(
                image
            )

            image.flags.writeable = True


            # =================================================
            # ถ้าพบโครงร่าง
            # =================================================

            if results.pose_landmarks:

                landmarks = (
                    results.pose_landmarks.landmark
                )


                # =================================================
                # LEFT SIDE
                # =================================================

                l_shoulder = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ].y
                ]

                l_hip = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_HIP.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.LEFT_HIP.value
                    ].y
                ]

                l_knee = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_KNEE.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.LEFT_KNEE.value
                    ].y
                ]

                l_ankle = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_ANKLE.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.LEFT_ANKLE.value
                    ].y
                ]

                l_foot = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
                    ].y
                ]

                l_heel = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_HEEL.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.LEFT_HEEL.value
                    ].y
                ]


                # =================================================
                # RIGHT SIDE
                # =================================================

                r_shoulder = [
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    ].y
                ]

                r_hip = [
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_HIP.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_HIP.value
                    ].y
                ]

                r_knee = [
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_KNEE.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_KNEE.value
                    ].y
                ]

                r_ankle = [
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_ANKLE.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_ANKLE.value
                    ].y
                ]

                r_foot = [
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
                    ].y
                ]

                r_heel = [
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_HEEL.value
                    ].x,

                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_HEEL.value
                    ].y
                ]


                # =================================================
                # คำนวณมุมเชิงคลินิกใน sagittal plane
                # =================================================

                left_knee_angle = clinical_knee_flexion(
                    l_hip,
                    l_knee,
                    l_ankle
                )

                right_knee_angle = clinical_knee_flexion(
                    r_hip,
                    r_knee,
                    r_ankle
                )

                left_hip_angle = clinical_hip_flexion(
                    l_shoulder,
                    l_hip,
                    l_knee,
                    direction_sign
                )

                right_hip_angle = clinical_hip_flexion(
                    r_shoulder,
                    r_hip,
                    r_knee,
                    direction_sign
                )

                left_ankle_angle = clinical_ankle_dorsiflexion(
                    l_knee,
                    l_ankle,
                    l_foot
                )

                right_ankle_angle = clinical_ankle_dorsiflexion(
                    r_knee,
                    r_ankle,
                    r_foot
                )

                mid_hip_x = (
                    l_hip[0] + r_hip[0]
                ) / 2.0

                left_heel_forward = (
                    l_heel[0] - mid_hip_x
                ) * direction_sign

                right_heel_forward = (
                    r_heel[0] - mid_hip_x
                ) * direction_sign


                # =================================================
                # เก็บข้อมูล
                # =================================================

                frames_data.append({

                    "Frame": frame_count,

                    "Time (s)": (
                        frame_count / fps
                    ),

                    "Left Hip Flexion":
                        left_hip_angle,

                    "Right Hip Flexion":
                        right_hip_angle,

                    "Left Knee Flexion":
                        left_knee_angle,

                    "Right Knee Flexion":
                        right_knee_angle,

                    "Left Ankle DF":
                        left_ankle_angle,

                    "Right Ankle DF":
                        right_ankle_angle,

                    "Left Heel Forward":
                        left_heel_forward,

                    "Right Heel Forward":
                        right_heel_forward
                })


                # =================================================
                # วาด Skeleton
                # =================================================

                mp_drawing.draw_landmarks(

                    image,

                    results.pose_landmarks,

                    mp_pose.POSE_CONNECTIONS,

                    landmark_drawing_spec=(
                        mp_drawing_styles
                        .get_default_pose_landmarks_style()
                    )
                )

                # เก็บเฟรมล่าสุดที่ตรวจพบโครงร่างสำหรับ Dashboard
                last_pose_image = image.copy()


            # =================================================
            # Progress
            # =================================================

            progress = min(
                frame_count / total_frames,
                1.0
            )

            progress_bar.progress(
                progress
            )

            status_text.text(
                f"กำลังวิเคราะห์เฟรม "
                f"{frame_count:,}/{total_frames:,}"
            )


            # -------------------------------------------------
            # แสดงทุก 2 Frame
            # -------------------------------------------------

            if frame_count % 2 == 0:

                st_frame.image(
                    image,
                    channels="RGB",
                    use_container_width=True
                )


    # =====================================================
    # ปิด Video
    # =====================================================

    cap.release()


    try:

        os.remove(
            video_path
        )

    except Exception:

        pass


    progress_bar.empty()

    status_text.empty()


    # =====================================================
    # วิเคราะห์ผล
    # =====================================================

    if not frames_data:

        st.error(
            "❌ ไม่พบโครงร่างร่างกายจากวิดีโอ"
        )

        st.info(
            "ลองใช้วิดีโอที่เห็นร่างกายเต็มตัว "
            "แสงเพียงพอ และไม่มีสิ่งกีดขวาง"
        )

    else:

        st.success(
            f"✅ วิเคราะห์เสร็จสิ้น "
            f"ตรวจพบข้อมูลจำนวน {len(frames_data):,} เฟรม"
        )


        # =================================================
        # DataFrame
        # =================================================

        df = pd.DataFrame(
            frames_data
        )


        # =================================================
        # ส่วนที่ 1: Medical AI Dashboard
        # =================================================

        cycle_analysis = build_gait_cycle_analysis(
            df,
            fps
        )

        screening = calculate_gait_screening(
            df,
            cycle_analysis
        )

        st.divider()

        st.markdown(
            '<div class="section-title">🩺 Medical Gait Dashboard</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'สรุปผลจากข้อมูลการเคลื่อนไหวที่ตรวจจับได้จากวิดีโอ '
            'ผลนี้เป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์'
            '</div>',
            unsafe_allow_html=True
        )

        left_col, center_col, right_col = st.columns(
            [1.05, 1.35, 1.05],
            gap="medium"
        )

        # -------------------------------------------------
        # LEFT: Gait Analysis
        # -------------------------------------------------
        with left_col:

            st.markdown(
                """
                <div class="med-card">
                    <div class="card-title">Gait Analysis</div>
                    <div class="small-muted">
                        ภาพโครงร่างจากเฟรมล่าสุดที่ระบบตรวจจับได้
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if last_pose_image is not None:
                st.image(
                    last_pose_image,
                    channels="RGB",
                    use_container_width=True
                )
            else:
                st.info("ไม่พบภาพโครงร่างสำหรับแสดงผล")

            if screening["cycle_available"]:

                st.metric(
                    "Hip Curve MAE",
                    f"{screening['hip_curve_mae']:.2f}°"
                )

                st.metric(
                    "Knee Curve MAE",
                    f"{screening['knee_curve_mae']:.2f}°"
                )

                st.metric(
                    "Ankle Curve MAE",
                    f"{screening['ankle_curve_mae']:.2f}°"
                )

            else:

                st.metric(
                    "Hip ROM SI",
                    f"{screening['hip_rom_si']:.2f}%"
                )

                st.metric(
                    "Knee ROM SI",
                    f"{screening['knee_rom_si']:.2f}%"
                )

                st.metric(
                    "Ankle ROM SI",
                    f"{screening['ankle_rom_si']:.2f}%"
                )

        # -------------------------------------------------
        # CENTER: Symmetry Score + Stability chart
        # -------------------------------------------------
        with center_col:

            curve_mae_caption = (
                f"{screening['overall_curve_mae']:.2f}°"
                if np.isfinite(
                    screening["overall_curve_mae"]
                )
                else "N/A"
            )

            phase_mae_caption = (
                f"{screening['phase_mae']:.2f}°"
                if np.isfinite(
                    screening["phase_mae"]
                )
                else "N/A"
            )

            st.markdown(
                f"""
                <div class="med-card">
                    <div class="card-title">Gait Screening Score</div>
                    <div class="score-number">
                        {screening['score']:.0f}%
                    </div>
                    <div class="screening-status">
                        {screening['status']}
                    </div>
                    <div class="score-caption">
                        Curve MAE {curve_mae_caption} ·
                        ROM SI {screening['overall_rom_si']:.2f}% ·
                        Phase diff {phase_mae_caption}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=screening["score"],
                    number={
                        "suffix": "%",
                        "font": {"size": 48}
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#94a3b8"
                        },
                        "bar": {
                            "color": "#22d3ee",
                            "thickness": 0.22
                        },
                        "bgcolor": "rgba(125,211,252,0.08)",
                        "borderwidth": 0,
                        "steps": [
                            {
                                "range": [0, 70],
                                "color": "rgba(239,68,68,0.40)"
                            },
                            {
                                "range": [70, 85],
                                "color": "rgba(245,158,11,0.42)"
                            },
                            {
                                "range": [85, 100],
                                "color": "rgba(34,197,94,0.42)"
                            }
                        ]
                    }
                )
            )

            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc")
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            if screening["cycle_available"]:

                joint_curve_df = pd.DataFrame({
                    "Joint": [
                        "Hip",
                        "Knee",
                        "Ankle"
                    ],
                    "Curve MAE (°)": [
                        screening[
                            "hip_curve_mae"
                        ],
                        screening[
                            "knee_curve_mae"
                        ],
                        screening[
                            "ankle_curve_mae"
                        ]
                    ]
                })

                fig_joint_si = px.bar(
                    joint_curve_df,
                    x="Joint",
                    y="Curve MAE (°)",
                    text_auto=".1f",
                    title=(
                        "Mean Left–Right "
                        "Joint Curve Difference"
                    )
                )

                fig_joint_si.update_layout(
                    height=285,
                    margin=dict(
                        l=20,
                        r=20,
                        t=55,
                        b=25
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor=(
                        "rgba(52, 92, 125, 0.18)"
                    ),
                    font=dict(
                        color="#e2e8f0"
                    ),
                    title_font=dict(
                        color="#f8fafc"
                    ),
                    xaxis=dict(
                        title="",
                        gridcolor=(
                            "rgba(148,163,184,0.10)"
                        )
                    ),
                    yaxis=dict(
                        title="Difference (°)",
                        gridcolor=(
                            "rgba(148,163,184,0.10)"
                        )
                    ),
                    showlegend=False
                )

                fig_joint_si.update_traces(
                    marker_color="#22d3ee"
                )

                st.plotly_chart(
                    fig_joint_si,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    }
                )

            else:

                st.info(
                    "ยังไม่แสดง Curve MAE "
                    "เพราะระบบตรวจ gait cycle "
                    "ได้ไม่เพียงพอ"
                )

        # -------------------------------------------------
        # RIGHT: Recommendations
        # -------------------------------------------------
        with right_col:

            st.markdown(
                """
                <div class="med-card">
                    <div class="card-title">Recommendations</div>
                    <div class="small-muted">
                        คำแนะนำจากค่าความสมมาตรที่ระบบคำนวณได้
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            def show_joint_status(
                label,
                curve_mae,
                rom_si,
                tolerance_deg
            ):

                if np.isfinite(curve_mae):

                    joint_score = (
                        0.65
                        * mae_to_score(
                            curve_mae,
                            tolerance_deg
                        )
                        + 0.35
                        * rom_si_to_score(
                            rom_si,
                            ROM_SI_TOLERANCE_PERCENT
                        )
                    )

                    detail = (
                        f"Curve MAE {curve_mae:.2f}° · "
                        f"ROM SI {rom_si:.2f}%"
                    )

                else:

                    joint_score = (
                        rom_si_to_score(
                            rom_si,
                            ROM_SI_TOLERANCE_PERCENT
                        )
                    )

                    detail = (
                        f"ROM SI {rom_si:.2f}% "
                        "(ยังไม่มี gait cycle)"
                    )

                if joint_score >= 80:

                    css_class = "status-good"
                    symbol = "✓"
                    message = "ความสมมาตรอยู่ในระดับดี"

                elif joint_score >= 60:

                    css_class = "status-watch"
                    symbol = "△"
                    message = "พบความแตกต่างบางส่วน"

                else:

                    css_class = "status-alert"
                    symbol = "⚠"
                    message = "พบความแตกต่างค่อนข้างมาก"

                st.markdown(
                    f'<div class="{css_class}">'
                    f'{symbol} <b>{label}</b><br>'
                    f'{detail}<br>'
                    f'{message}</div>',
                    unsafe_allow_html=True
                )

            show_joint_status(
                "Hip",
                screening[
                    "hip_curve_mae"
                ],
                screening[
                    "hip_rom_si"
                ],
                CURVE_MAE_TOLERANCE_DEG[
                    "Hip"
                ]
            )

            show_joint_status(
                "Knee",
                screening[
                    "knee_curve_mae"
                ],
                screening[
                    "knee_rom_si"
                ],
                CURVE_MAE_TOLERANCE_DEG[
                    "Knee"
                ]
            )

            show_joint_status(
                "Ankle",
                screening[
                    "ankle_curve_mae"
                ],
                screening[
                    "ankle_rom_si"
                ],
                CURVE_MAE_TOLERANCE_DEG[
                    "Ankle"
                ]
            )

            st.markdown(
                f"""
                <div class="med-card">
                    <div class="card-title">คำแนะนำโดยรวม</div>
                    <div class="small-muted">
                        {screening['recommendation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="status-watch">
                    ⚠ <b>ข้อควรระวัง</b><br>
                    ผลลัพธ์นี้เป็นการคัดกรองจากวิดีโอ 2D และอาจได้รับ
                    ผลกระทบจากมุมกล้อง แสง เสื้อผ้า การบังร่างกาย
                    และคุณภาพวิดีโอ
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # Summary metrics
        # -------------------------------------------------
        st.markdown(
            '<div class="section-title">📊 ตัวชี้วัดหลัก</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            if np.isfinite(
                screening["overall_curve_mae"]
            ):
                st.metric(
                    "Overall Curve MAE",
                    f"{screening['overall_curve_mae']:.2f}°"
                )
            else:
                st.metric(
                    "Overall Curve MAE",
                    "N/A"
                )

        with m2:
            st.metric(
                "Overall ROM SI",
                f"{screening['overall_rom_si']:.2f}%"
            )

        with m3:
            st.metric(
                "Frames วิเคราะห์",
                f"{len(df):,}"
            )

        with m4:
            st.metric(
                "ระยะเวลาวิเคราะห์",
                f"{df['Time (s)'].max():.1f} s"
            )


        # =================================================
        # Gait Cycle 0-100%
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">🔄 Gait Cycle Analysis · 0–100%</div>',
            unsafe_allow_html=True
        )

        if (
            cycle_analysis is not None
            and cycle_analysis.get(
                "available",
                False
            )
        ):

            left_cycle = cycle_analysis[
                "left_cycle"
            ]

            right_cycle = cycle_analysis[
                "right_cycle"
            ]

            gait_percent = left_cycle[
                "percent"
            ]

            lc = left_cycle[
                "mean_curves"
            ]

            rc = right_cycle[
                "mean_curves"
            ]

            cycle_c1, cycle_c2, cycle_c3 = st.columns(3)

            with cycle_c1:
                st.metric(
                    "Left gait cycles",
                    left_cycle["cycle_count"]
                )

            with cycle_c2:
                st.metric(
                    "Right gait cycles",
                    right_cycle["cycle_count"]
                )

            with cycle_c3:
                mean_cycle_duration = np.mean([
                    left_cycle[
                        "mean_cycle_duration"
                    ],
                    right_cycle[
                        "mean_cycle_duration"
                    ],
                ])

                st.metric(
                    "Mean stride time",
                    f"{mean_cycle_duration:.2f} s"
                )

            def make_cycle_fig(
                percent,
                left_curve,
                right_curve,
                title,
                y_title
            ):

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=percent,
                        y=left_curve,
                        mode="lines",
                        name="Left",
                        line=dict(
                            color="#22d3ee",
                            width=3
                        )
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=percent,
                        y=right_curve,
                        mode="lines",
                        name="Right",
                        line=dict(
                            color="#c084fc",
                            width=3
                        )
                    )
                )

                phase_colors = [
                    "rgba(34,211,238,0.04)",
                    "rgba(34,211,238,0.08)",
                    "rgba(96,165,250,0.05)",
                    "rgba(96,165,250,0.09)",
                    "rgba(192,132,252,0.05)",
                    "rgba(192,132,252,0.09)",
                    "rgba(167,139,250,0.05)",
                    "rgba(167,139,250,0.09)",
                ]

                for (
                    phase,
                    start_p,
                    end_p
                ), shade in zip(
                    GAIT_PHASES,
                    phase_colors
                ):

                    fig.add_vrect(
                        x0=start_p,
                        x1=end_p,
                        fillcolor=shade,
                        line_width=0,
                        layer="below"
                    )

                for boundary in [
                    10,
                    30,
                    50,
                    60,
                    73,
                    87
                ]:

                    fig.add_vline(
                        x=boundary,
                        line_width=1,
                        line_dash="dot",
                        line_color=(
                            "rgba(203,213,225,0.25)"
                        )
                    )

                fig.update_layout(
                    title=title,
                    xaxis_title="% Gait Cycle",
                    yaxis_title=y_title,
                    hovermode="x unified",
                    height=430,
                    margin=dict(
                        l=35,
                        r=20,
                        t=60,
                        b=40
                    ),
                    paper_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),
                    plot_bgcolor=(
                        "rgba(52,92,125,0.18)"
                    ),
                    font=dict(
                        color="#e2e8f0"
                    ),
                    legend_title_text=""
                )

                return fig

            gait_tab1, gait_tab2, gait_tab3 = st.tabs(
                [
                    "🦿 Hip",
                    "🦵 Knee",
                    "🦶 Ankle"
                ]
            )

            with gait_tab1:

                st.plotly_chart(
                    make_cycle_fig(
                        gait_percent,
                        lc[
                            "Left Hip Flexion"
                        ],
                        rc[
                            "Right Hip Flexion"
                        ],
                        (
                            "Hip angle across "
                            "normalized gait cycle"
                        ),
                        (
                            "Flexion (+) / "
                            "Extension (-) °"
                        )
                    ),
                    use_container_width=True
                )

            with gait_tab2:

                st.plotly_chart(
                    make_cycle_fig(
                        gait_percent,
                        lc[
                            "Left Knee Flexion"
                        ],
                        rc[
                            "Right Knee Flexion"
                        ],
                        (
                            "Knee flexion across "
                            "normalized gait cycle"
                        ),
                        "Knee Flexion (°)"
                    ),
                    use_container_width=True
                )

            with gait_tab3:

                st.plotly_chart(
                    make_cycle_fig(
                        gait_percent,
                        lc[
                            "Left Ankle DF"
                        ],
                        rc[
                            "Right Ankle DF"
                        ],
                        (
                            "Ankle angle across "
                            "normalized gait cycle"
                        ),
                        (
                            "Dorsiflexion (+) / "
                            "Plantarflexion (-) °"
                        )
                    ),
                    use_container_width=True
                )

            # ---------------------------------------------
            # Extract phase metrics
            # ---------------------------------------------

            left_metrics = extract_phase_metrics(
                gait_percent,
                lc[
                    "Left Hip Flexion"
                ],
                lc[
                    "Left Knee Flexion"
                ],
                lc[
                    "Left Ankle DF"
                ]
            )

            right_metrics = extract_phase_metrics(
                gait_percent,
                rc[
                    "Right Hip Flexion"
                ],
                rc[
                    "Right Knee Flexion"
                ],
                rc[
                    "Right Ankle DF"
                ]
            )

            reference_rows = []

            for metric_name in REFERENCE_BANDS:

                low, high = REFERENCE_BANDS[
                    metric_name
                ]

                left_value = left_metrics[
                    metric_name
                ]

                right_value = right_metrics[
                    metric_name
                ]

                reference_rows.append({
                    "ตัวชี้วัด": metric_name,
                    "ซ้าย (°)": left_value,
                    "ขวา (°)": right_value,
                    "ช่วงอ้างอิง (°)": (
                        f"{low:g} ถึง {high:g}"
                    ),
                    "ซ้าย": reference_status(
                        left_value,
                        metric_name
                    ),
                    "ขวา": reference_status(
                        right_value,
                        metric_name
                    ),
                })

            reference_df = pd.DataFrame(
                reference_rows
            )

            st.markdown(
                '<div class="section-title">'
                '📚 Phase-based reference comparison'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "ช่วงอ้างอิงใช้เพื่อช่วยอ่านกราฟ ไม่ถูกนำไปใช้วินิจฉัย "
                "และไม่ถูกใช้เป็นเกณฑ์ตัดสินคะแนนโดยตรง"
            )

            st.dataframe(
                reference_df.style.format({
                    "ซ้าย (°)": "{:.2f}",
                    "ขวา (°)": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True
            )

            # ---------------------------------------------
            # Gait phase table
            # ---------------------------------------------

            phase_df = pd.DataFrame(
                [
                    {
                        "Phase": phase,
                        "% Gait Cycle":
                            f"{start_p}–{end_p}%"
                    }
                    for (
                        phase,
                        start_p,
                        end_p
                    ) in GAIT_PHASES
                ]
            )

            with st.expander(
                "ℹ️ ดูช่วงของ 8 Gait Phases"
            ):

                st.dataframe(
                    phase_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown(
                    """
                    **หมายเหตุเรื่องข้อเท้า:** ในการเดินปกติ
                    ข้อเท้ามักเพิ่ม dorsiflexion ในช่วง terminal stance
                    แล้วจึง plantarflex อย่างรวดเร็วในช่วง pre-swing /
                    push-off ดังนั้นระบบนี้ไม่ตีความ
                    “terminal stance = plantarflexion” โดยตรง
                    """
                )

        else:

            st.warning(
                "ยังตรวจพบ heel-strike ต่อเนื่องไม่เพียงพอ "
                "จึงยังไม่สามารถสร้างกราฟ 0–100% gait cycle ได้"
            )

            st.info(
                "แนะนำให้ใช้วิดีโอด้านข้างที่เห็นส้นเท้าชัด "
                "มีอย่างน้อย 2–3 stride ต่อข้าง และกล้องนิ่ง"
            )


        # =================================================
        # ส่วนที่ 2: กราฟการเคลื่อนไหว
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📈 Joint Angle Over Time</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'เปรียบเทียบมุมข้อต่อด้านซ้ายและขวาตลอดช่วงเวลาของวิดีโอ'
            '</div>',
            unsafe_allow_html=True
        )

        # Knee
        fig_knee = px.line(
            df,
            x="Time (s)",
            y=[
                "Left Knee Flexion",
                "Right Knee Flexion"
            ],
            labels={
                "value": "มุม (องศา)",
                "variable": "ข้าง",
                "Time (s)": "เวลา (วินาที)"
            },
            title="Knee Flexion / Extension"
        )

        # Hip
        fig_hip = px.line(
            df,
            x="Time (s)",
            y=[
                "Left Hip Flexion",
                "Right Hip Flexion"
            ],
            labels={
                "value": "มุม (องศา)",
                "variable": "ข้าง",
                "Time (s)": "เวลา (วินาที)"
            },
            title="Hip Flexion (+) / Extension (-)"
        )

        # Ankle
        fig_ankle = px.line(
            df,
            x="Time (s)",
            y=[
                "Left Ankle DF",
                "Right Ankle DF"
            ],
            labels={
                "value": "มุม (องศา)",
                "variable": "ข้าง",
                "Time (s)": "เวลา (วินาที)"
            },
            title="Ankle Dorsiflexion (+) / Plantarflexion (-)"
        )

        for fig in [
            fig_knee,
            fig_hip,
            fig_ankle
        ]:
            fig.update_layout(
                hovermode="x unified",
                legend_title_text="",
                height=420,
                margin=dict(l=30, r=20, t=55, b=35),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(52, 92, 125, 0.18)",
                font=dict(color="#e2e8f0"),
                title_font=dict(color="#f8fafc"),
                xaxis=dict(
                    gridcolor="rgba(148,163,184,0.10)"
                ),
                yaxis=dict(
                    gridcolor="rgba(148,163,184,0.10)"
                )
            )

        # สีเส้นให้ตัดกับพื้นหลังและอ่านง่าย
        for fig in [fig_knee, fig_hip, fig_ankle]:
            if len(fig.data) >= 1:
                fig.data[0].line.color = "#22d3ee"
                fig.data[0].line.width = 3
            if len(fig.data) >= 2:
                fig.data[1].line.color = "#c084fc"
                fig.data[1].line.width = 3

        tab_knee, tab_hip, tab_ankle = st.tabs(
            [
                "🦵 Knee",
                "🦿 Hip",
                "🦶 Ankle"
            ]
        )

        with tab_knee:
            st.plotly_chart(
                fig_knee,
                use_container_width=True
            )

        with tab_hip:
            st.plotly_chart(
                fig_hip,
                use_container_width=True
            )

        with tab_ankle:
            st.plotly_chart(
                fig_ankle,
                use_container_width=True
            )


        # =================================================
        # ส่วนที่ 3: สรุป Knee
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">🦵 สรุปการเคลื่อนไหวของเข่า</div>',
            unsafe_allow_html=True
        )


        mean_left_knee = df[
            "Left Knee Flexion"
        ].mean()

        mean_right_knee = df[
            "Right Knee Flexion"
        ].mean()

        max_left_knee = df[
            "Left Knee Flexion"
        ].max()

        max_right_knee = df[
            "Right Knee Flexion"
        ].max()

        min_left_knee = df[
            "Left Knee Flexion"
        ].min()

        min_right_knee = df[
            "Right Knee Flexion"
        ].min()

        rom_left_knee = calculate_rom(
            df["Left Knee Flexion"]
        )

        rom_right_knee = calculate_rom(
            df["Right Knee Flexion"]
        )


        si_mean_knee = calculate_symmetry_index(
            mean_left_knee,
            mean_right_knee
        )

        si_max_knee = calculate_symmetry_index(
            max_left_knee,
            max_right_knee
        )


        k1, k2, k3, k4 = st.columns(4)


        with k1:

            st.metric(
                "เข่าซ้ายเฉลี่ย",
                f"{mean_left_knee:.2f}°"
            )


        with k2:

            st.metric(
                "เข่าขวาเฉลี่ย",
                f"{mean_right_knee:.2f}°"
            )


        with k3:

            st.metric(
                "ROM ซ้าย / ขวา",
                f"{rom_left_knee:.1f}° / "
                f"{rom_right_knee:.1f}°"
            )


        with k4:

            st.metric(
                "Knee SI",
                f"{si_mean_knee:.2f}%"
            )


        # =================================================
        # ส่วนที่ 4: SI รายข้อต่อ
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">🔎 ความแตกต่างซ้าย–ขวารายข้อต่อ</div>',
            unsafe_allow_html=True
        )


        si_df = pd.DataFrame({

            "ข้อต่อ": [
                "Hip",
                "Knee",
                "Ankle"
            ],

            "Curve MAE (°)": [
                screening[
                    "hip_curve_mae"
                ],
                screening[
                    "knee_curve_mae"
                ],
                screening[
                    "ankle_curve_mae"
                ]
            ],

            "ROM Symmetry Index (%)": [
                screening[
                    "hip_rom_si"
                ],
                screening[
                    "knee_rom_si"
                ],
                screening[
                    "ankle_rom_si"
                ]
            ]
        })



        st.dataframe(
            si_df.style.format({
                "Curve MAE (°)": "{:.2f}",
                "ROM Symmetry Index (%)": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # ส่วนที่ 5: Dashboard รายข้อต่อ
        # =================================================

        st.markdown(
            '<div class="section-title">🧩 รายละเอียดแต่ละข้อต่อ</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.markdown("### 🦿 Hip")

            if np.isfinite(
                screening["hip_curve_mae"]
            ):
                st.metric(
                    "Curve MAE",
                    f"{screening['hip_curve_mae']:.2f}°"
                )

            st.metric(
                "ROM SI",
                f"{screening['hip_rom_si']:.2f}%"
            )


        with c2:

            st.markdown("### 🦵 Knee")

            if np.isfinite(
                screening["knee_curve_mae"]
            ):
                st.metric(
                    "Curve MAE",
                    f"{screening['knee_curve_mae']:.2f}°"
                )

            st.metric(
                "ROM SI",
                f"{screening['knee_rom_si']:.2f}%"
            )


        with c3:

            st.markdown("### 🦶 Ankle")

            if np.isfinite(
                screening["ankle_curve_mae"]
            ):
                st.metric(
                    "Curve MAE",
                    f"{screening['ankle_curve_mae']:.2f}°"
                )

            st.metric(
                "ROM SI",
                f"{screening['ankle_rom_si']:.2f}%"
            )


        # =================================================
        # ส่วนที่ 6: ตารางสรุป
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📋 ตารางสรุปมุมข้อต่อ</div>',
            unsafe_allow_html=True
        )


        summary_data = {

            "ข้อต่อ": [
                "Hip",
                "Knee",
                "Ankle"
            ],

            "ซ้ายเฉลี่ย (°)": [

                df[
                    "Left Hip Flexion"
                ].mean(),

                df[
                    "Left Knee Flexion"
                ].mean(),

                df[
                    "Left Ankle DF"
                ].mean()
            ],

            "ขวาเฉลี่ย (°)": [

                df[
                    "Right Hip Flexion"
                ].mean(),

                df[
                    "Right Knee Flexion"
                ].mean(),

                df[
                    "Right Ankle DF"
                ].mean()
            ],

            "ซ้าย ROM (°)": [

                calculate_rom(
                    df["Left Hip Flexion"]
                ),

                calculate_rom(
                    df["Left Knee Flexion"]
                ),

                calculate_rom(
                    df["Left Ankle DF"]
                )
            ],

            "ขวา ROM (°)": [

                calculate_rom(
                    df["Right Hip Flexion"]
                ),

                calculate_rom(
                    df["Right Knee Flexion"]
                ),

                calculate_rom(
                    df["Right Ankle DF"]
                )
            ]
        }


        summary_df = pd.DataFrame(
            summary_data
        )


        st.dataframe(
            summary_df.style.format({
                "ซ้ายเฉลี่ย (°)": "{:.2f}",
                "ขวาเฉลี่ย (°)": "{:.2f}",
                "ซ้าย ROM (°)": "{:.2f}",
                "ขวา ROM (°)": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # ส่วนที่ 7: เกณฑ์การแปลผล
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📖 เกณฑ์การแปลผล</div>',
            unsafe_allow_html=True
        )


        interpretation_df = pd.DataFrame({

            "ระดับ": [
                "🟢 ปกติ",
                "🟡 ควรประเมินเพิ่มเติม",
                "🔴 พบความแตกต่างมาก"
            ],

            "Overall SI": [
                "< 5%",
                "5% – < 10%",
                "≥ 10%"
            ],

            "ความหมาย": [

                "ความแตกต่างซ้าย–ขวาอยู่ในระดับต่ำ",

                "มีความแตกต่างระดับปานกลาง",

                "มีความแตกต่างค่อนข้างมาก"
            ]
        })


        st.dataframe(
            interpretation_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # ส่วนที่ 8: คำเตือน
        # =================================================

        st.warning(
            """
            ⚠️ **คำเตือนสำคัญ**

            Gait Screening Score และเกณฑ์สีในระบบนี้
            ใช้สำหรับการคัดกรองเบื้องต้นจากข้อมูลวิดีโอและ
            Symmetry Index เท่านั้น

            ไม่สามารถใช้ยืนยันหรือวินิจฉัยโรค
            หรือความผิดปกติทางการแพทย์ได้

            ผลลัพธ์อาจได้รับผลกระทบจากมุมกล้อง แสง เสื้อผ้า
            การบังส่วนต่าง ๆ ของร่างกาย คุณภาพวิดีโอ
            และความแม่นยำของระบบตรวจจับท่าทาง
            """
        )


        # =================================================
        # ส่วนที่ 9: ดาวน์โหลดข้อมูล
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📥 ส่งออกข้อมูล</div>',
            unsafe_allow_html=True
        )


        csv_data = df.to_csv(
            index=False
        ).encode("utf-8-sig")


        st.download_button(

            label="📥 ดาวน์โหลดข้อมูลทั้งหมด (.CSV)",

            data=csv_data,

            file_name="gait_analysis_data.csv",

            mime="text/csv",

            use_container_width=True
        )


        st.divider()

        st.markdown(
            '<div class="section-title">🧾 ข้อมูลประกอบการประเมิน</div>',
            unsafe_allow_html=True
        )

        context_c1, context_c2, context_c3 = st.columns(3)

        with context_c1:
            st.metric(
                "อายุ",
                f"{subject_age} ปี"
            )

        with context_c2:
            st.metric(
                "เพศในรายงาน",
                subject_sex
            )

        with context_c3:
            st.metric(
                "ทิศทาง",
                walking_direction
            )

        st.info(
            "อายุ เพศ ความเร็วเดิน และสัดส่วนร่างกายสามารถมีผลต่อ gait kinematics "
            "แต่เวอร์ชันนี้ไม่ได้ใช้เพศหรืออายุเป็นตัวปรับคะแนนหรือเป็นตัวตัดสิน "
            "ความผิดปกติโดยอัตโนมัติ"
        )


        # =================================================
        # Footer
        # =================================================

        st.markdown(
            """
            <div class="footer-note">

            🦶 **Video Gait Analysis System**

            ระบบนี้จัดทำขึ้นเพื่อการศึกษาและการคัดกรองเบื้องต้น
            ผลลัพธ์ควรพิจารณาร่วมกับการสังเกตทางคลินิก
            และการประเมินโดยผู้เชี่ยวชาญเมื่อจำเป็น

            </div>
            """,
            unsafe_allow_html=True
        )
