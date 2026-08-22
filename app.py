import os
import tempfile
from pathlib import Path

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


# =========================================================
# 2. ตั้งค่าหน้าเว็บ
# =========================================================

st.set_page_config(
    page_title="Medical Gait AI | Gait Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 3. CSS สำหรับ UI - Dark Medical AI Dashboard
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg-0: #071525;
        --bg-1: #0b2035;
        --bg-2: #10304d;
        --panel: rgba(15, 42, 67, 0.88);
        --panel-strong: rgba(17, 50, 79, 0.96);
        --panel-soft: rgba(20, 58, 91, 0.62);
        --line: rgba(125, 211, 252, 0.16);
        --line-strong: rgba(103, 232, 249, 0.30);
        --cyan: #22d3ee;
        --cyan-soft: #67e8f9;
        --blue: #60a5fa;
        --violet: #a78bfa;
        --green: #34d399;
        --yellow: #facc15;
        --orange: #fb923c;
        --red: #fb7185;
        --text: #f8fafc;
        --muted: #a9bfd2;
        --muted-2: #7894ac;
    }

    html, body, [class*="css"] {
        font-family:
            Inter, ui-sans-serif, system-ui, -apple-system,
            BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(
                circle at 12% 0%,
                rgba(34, 211, 238, 0.15),
                transparent 27%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(167, 139, 250, 0.12),
                transparent 24%
            ),
            linear-gradient(
                180deg,
                #0a2035 0%,
                #08192a 46%,
                #061321 100%
            );
    }

    /* subtle technology grid made with CSS only */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.13;
        background-image:
            linear-gradient(
                rgba(125, 211, 252, 0.10) 1px,
                transparent 1px
            ),
            linear-gradient(
                90deg,
                rgba(125, 211, 252, 0.10) 1px,
                transparent 1px
            );
        background-size: 46px 46px;
        mask-image: linear-gradient(
            to bottom,
            rgba(0,0,0,0.70),
            rgba(0,0,0,0.10)
        );
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1540px;
        padding-top: 1.1rem;
        padding-bottom: 3.5rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* --------------------------------------------------
       Top product bar
       -------------------------------------------------- */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        min-height: 68px;
        padding: 14px 20px;
        margin-bottom: 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background:
            linear-gradient(
                135deg,
                rgba(17, 51, 80, 0.94),
                rgba(10, 31, 51, 0.92)
            );
        box-shadow:
            0 18px 50px rgba(1, 10, 22, 0.28),
            inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(18px);
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        color: #03131d;
        font-size: 13px;
        font-weight: 900;
        letter-spacing: -0.4px;
        background:
            linear-gradient(
                135deg,
                var(--cyan-soft),
                var(--blue)
            );
        box-shadow:
            0 0 24px rgba(34, 211, 238, 0.25);
    }

    .brand {
        color: #f8fbff;
        font-size: 21px;
        line-height: 1.05;
        font-weight: 820;
        letter-spacing: -0.3px;
    }

    .brand-sub {
        color: var(--muted-2);
        font-size: 10px;
        margin-top: 4px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 750;
    }

    .system-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 11px;
        border-radius: 999px;
        border: 1px solid rgba(52, 211, 153, 0.24);
        color: #c7f9e4;
        background: rgba(16, 185, 129, 0.08);
        font-size: 11px;
        font-weight: 760;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .system-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 14px rgba(52, 211, 153, 0.75);
    }

    /* --------------------------------------------------
       Hero — text / telemetry only, no image
       -------------------------------------------------- */
    .hero-panel {
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.75fr);
        gap: 26px;
        align-items: stretch;
        padding: 32px;
        margin-bottom: 18px;
        border-radius: 22px;
        border: 1px solid rgba(103, 232, 249, 0.20);
        background:
            linear-gradient(
                125deg,
                rgba(22, 66, 101, 0.93),
                rgba(14, 42, 67, 0.94) 58%,
                rgba(18, 39, 68, 0.94)
            );
        box-shadow:
            0 22px 60px rgba(0, 8, 20, 0.28),
            inset 0 1px 0 rgba(255,255,255,0.06);
    }

    .hero-panel::before {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        left: -170px;
        top: -270px;
        border-radius: 50%;
        background:
            radial-gradient(
                circle,
                rgba(34, 211, 238, 0.18),
                transparent 68%
            );
    }

    .hero-main,
    .telemetry-panel {
        position: relative;
        z-index: 1;
    }

    .hero-kicker {
        color: var(--cyan-soft);
        font-size: 11px;
        font-weight: 840;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        max-width: 820px;
        color: #ffffff;
        font-size: clamp(31px, 3.2vw, 48px);
        font-weight: 860;
        line-height: 1.06;
        letter-spacing: -1.15px;
        margin-bottom: 14px;
    }

    .hero-copy {
        max-width: 830px;
        color: #c4d7e7;
        font-size: 14px;
        line-height: 1.75;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 20px;
    }

    .hero-tag {
        padding: 7px 10px;
        border-radius: 8px;
        border: 1px solid rgba(125, 211, 252, 0.17);
        background: rgba(125, 211, 252, 0.06);
        color: #d7edf8;
        font-size: 10px;
        font-weight: 760;
        letter-spacing: 0.45px;
        text-transform: uppercase;
    }

    .telemetry-panel {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 9px;
        padding: 15px;
        border-radius: 16px;
        border: 1px solid rgba(125, 211, 252, 0.14);
        background: rgba(5, 23, 39, 0.32);
    }

    .telemetry-title {
        color: #7dd3fc;
        font-size: 10px;
        font-weight: 820;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin: 0 0 4px 2px;
    }

    .telemetry-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 11px 12px;
        border-radius: 11px;
        background: rgba(18, 55, 84, 0.55);
        border: 1px solid rgba(148, 197, 224, 0.09);
    }

    .telemetry-label {
        color: var(--muted-2);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 760;
    }

    .telemetry-value {
        color: #f1f8fd;
        font-size: 11px;
        font-weight: 790;
        text-align: right;
    }

    /* --------------------------------------------------
       Section headings
       -------------------------------------------------- */
    .section-title {
        position: relative;
        padding-left: 12px;
        color: #f6fbff;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.1px;
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
    }

    .section-title::before {
        content: "";
        position: absolute;
        left: 0;
        top: 3px;
        bottom: 3px;
        width: 3px;
        border-radius: 999px;
        background:
            linear-gradient(
                180deg,
                var(--cyan),
                var(--violet)
            );
        box-shadow: 0 0 15px rgba(34, 211, 238, 0.35);
    }

    .section-subtitle {
        color: var(--muted);
        margin-bottom: 1rem;
        font-size: 0.88rem;
        line-height: 1.65;
    }

    /* --------------------------------------------------
       Cards
       -------------------------------------------------- */
    .med-card {
        position: relative;
        overflow: hidden;
        padding: 19px;
        margin-bottom: 14px;
        border-radius: 16px;
        border: 1px solid rgba(125, 211, 252, 0.13);
        background:
            linear-gradient(
                145deg,
                rgba(19, 57, 88, 0.88),
                rgba(9, 31, 51, 0.92)
            );
        box-shadow:
            0 14px 34px rgba(1, 10, 22, 0.20),
            inset 0 1px 0 rgba(255,255,255,0.035);
    }

    .med-card::after {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 2px;
        width: 38%;
        background:
            linear-gradient(
                90deg,
                rgba(34, 211, 238, 0.75),
                transparent
            );
    }

    .card-title {
        color: #f8fbff;
        font-size: 16px;
        font-weight: 790;
        padding-bottom: 10px;
        margin-bottom: 13px;
        border-bottom: 1px solid rgba(148, 197, 224, 0.10);
    }

    .small-muted {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.65;
    }

    .score-number {
        color: #ffffff;
        font-size: 62px;
        line-height: 1;
        font-weight: 900;
        text-align: center;
        letter-spacing: -2.2px;
        margin: 13px 0 8px;
        text-shadow: 0 0 28px rgba(34, 211, 238, 0.12);
    }

    .score-unit {
        margin-left: 6px;
        color: #7897ae;
        font-size: 18px;
        font-weight: 720;
        letter-spacing: 0;
    }

    .screening-status {
        text-align: center;
        font-size: 15px;
        font-weight: 790;
        margin: 10px 0 5px;
    }

    .score-status-normal { color: #6ee7b7; }
    .score-status-mild { color: #fde047; }
    .score-status-warning { color: #fdba74; }
    .score-status-danger { color: #fda4af; }

    .score-caption {
        color: var(--muted);
        text-align: center;
        font-size: 11px;
        line-height: 1.65;
        margin-top: 8px;
    }

    .model-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 9px;
        margin: 12px 0 18px;
    }

    .model-cell {
        padding: 11px 12px;
        border-radius: 11px;
        border: 1px solid rgba(125, 211, 252, 0.11);
        background: rgba(15, 48, 74, 0.56);
    }

    .model-weight {
        color: #7dd3fc;
        font-size: 18px;
        font-weight: 850;
        line-height: 1;
    }

    .model-label {
        color: var(--muted-2);
        font-size: 9px;
        font-weight: 760;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* --------------------------------------------------
       Joint status cards
       -------------------------------------------------- */
    .status-good,
    .status-mild,
    .status-watch,
    .status-alert {
        padding: 13px 14px;
        border-radius: 12px;
        margin-bottom: 9px;
        line-height: 1.55;
        border: 1px solid;
        font-size: 12px;
    }

    .status-good {
        color: #d1fae5;
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(52, 211, 153, 0.23);
    }

    .status-mild {
        color: #fef9c3;
        background: rgba(250, 204, 21, 0.07);
        border-color: rgba(250, 204, 21, 0.22);
    }

    .status-watch {
        color: #ffedd5;
        background: rgba(251, 146, 60, 0.08);
        border-color: rgba(251, 146, 60, 0.24);
    }

    .status-alert {
        color: #ffe4e6;
        background: rgba(251, 113, 133, 0.08);
        border-color: rgba(251, 113, 133, 0.23);
    }

    /* --------------------------------------------------
       Streamlit widgets
       -------------------------------------------------- */
    [data-testid="stMetric"] {
        min-height: 95px;
        padding: 14px 15px;
        border-radius: 14px;
        border: 1px solid rgba(125, 211, 252, 0.12);
        background:
            linear-gradient(
                145deg,
                rgba(20, 58, 89, 0.84),
                rgba(9, 30, 49, 0.90)
            );
        box-shadow: 0 12px 28px rgba(0,0,0,0.16);
    }

    [data-testid="stMetricLabel"] {
        color: #8facbf;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: #f4faff;
        font-weight: 820;
    }

    [data-testid="stFileUploader"] {
        padding: 12px;
        border-radius: 16px;
        border: 1px dashed rgba(34, 211, 238, 0.40);
        background: rgba(18, 55, 84, 0.62);
    }

    [data-testid="stFileUploader"] section {
        border-radius: 12px;
        background: rgba(13, 42, 66, 0.70);
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
        border: 1px solid rgba(148, 197, 224, 0.13);
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border-radius: 14px;
        border: 1px solid rgba(125, 211, 252, 0.11);
    }

    div[data-baseweb="tab-list"] {
        gap: 7px;
        padding: 5px;
        border-radius: 12px;
        background: rgba(13, 43, 68, 0.70);
        border: 1px solid rgba(125, 211, 252, 0.08);
    }

    button[data-baseweb="tab"] {
        color: #8eaabe;
        border-radius: 9px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fbff;
        background: rgba(34, 211, 238, 0.08);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0c263e,
                #081a2b
            );
        border-right: 1px solid rgba(125, 211, 252, 0.10);
    }

    .stButton button,
    .stDownloadButton button {
        min-height: 45px;
        border-radius: 11px;
        border: 1px solid rgba(34, 211, 238, 0.40);
        color: #f8fbff;
        font-weight: 760;
        background:
            linear-gradient(
                90deg,
                rgba(2, 132, 199, 0.92),
                rgba(8, 145, 178, 0.92)
            );
        box-shadow: 0 9px 24px rgba(2, 132, 199, 0.13);
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        color: #ffffff;
        border-color: #67e8f9;
        box-shadow: 0 10px 28px rgba(34, 211, 238, 0.18);
    }

    hr {
        border-color: rgba(148, 197, 224, 0.10) !important;
    }

    .footer-note {
        margin-top: 22px;
        padding: 18px 3px 8px;
        color: #7894ac;
        font-size: 0.78rem;
        line-height: 1.7;
        border-top: 1px solid rgba(148, 197, 224, 0.10);
    }

    @media (max-width: 980px) {
        .hero-panel {
            grid-template-columns: 1fr;
            padding: 24px;
        }

        .telemetry-panel {
            display: grid;
            grid-template-columns: 1fr;
        }

        .model-strip {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 0.7rem;
        }

        .top-nav {
            padding: 12px 14px;
        }

        .system-badge {
            display: none;
        }

        .brand {
            font-size: 18px;
        }

        .hero-title {
            font-size: 30px;
        }

        .score-number {
            font-size: 50px;
        }
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
        <div class="brand-wrap">
            <div class="brand-mark">AI</div>
            <div>
                <div class="brand">Medical Gait AI</div>
                <div class="brand-sub">Markerless Gait Screening Platform</div>
            </div>
        </div>

        <div class="system-badge">
            <span class="system-dot"></span>
            Research Prototype
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-main">
            <div class="hero-kicker">AI-Assisted Movement Analytics</div>
            <div class="hero-title">
                Video Gait Analysis<br>
                for Clinical Screening
            </div>
            <div class="hero-copy">
                วิเคราะห์การเดินจากวิดีโอด้านข้างด้วย 2D pose estimation
                พร้อมสร้าง normalized gait cycle 0–100%,
                คำนวณ Hip / Knee / Ankle kinematics,
                Curve MAE, ROM Symmetry และ Phase-based metrics
                เพื่อช่วยคัดกรองและติดตามความเปลี่ยนแปลงอย่างเป็นระบบ
            </div>

            <div class="hero-tags">
                <span class="hero-tag">2D Pose Tracking</span>
                <span class="hero-tag">Gait Cycle 0–100%</span>
                <span class="hero-tag">Curve MAE</span>
                <span class="hero-tag">ROM Symmetry</span>
                <span class="hero-tag">Phase Metrics</span>
            </div>
        </div>

        <div class="telemetry-panel">
            <div class="telemetry-title">Analysis Pipeline</div>

            <div class="telemetry-row">
                <span class="telemetry-label">Pose Engine</span>
                <span class="telemetry-value">MediaPipe Pose</span>
            </div>

            <div class="telemetry-row">
                <span class="telemetry-label">Plane</span>
                <span class="telemetry-value">Sagittal / 2D</span>
            </div>

            <div class="telemetry-row">
                <span class="telemetry-label">Cycle Model</span>
                <span class="telemetry-value">Estimated IC / Stride Anchor</span>
            </div>

            <div class="telemetry-row">
                <span class="telemetry-label">Score Model</span>
                <span class="telemetry-value">40 / 35 / 25</span>
            </div>
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
# 5.1 Lower-limb Joint Tracking Visualization
# =========================================================

def normalized_to_pixel(point, image_shape):
    """Convert MediaPipe normalized (x, y) coordinates to image pixels."""
    height, width = image_shape[:2]
    x = int(np.clip(point[0], 0.0, 1.0) * (width - 1))
    y = int(np.clip(point[1], 0.0, 1.0) * (height - 1))
    return x, y


def draw_text_box(image, text, origin, color, font_scale=0.48):
    """Draw a compact high-contrast label on an RGB frame."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    height, width = image.shape[:2]
    x = int(np.clip(x, 4, max(4, width - text_w - 10)))
    y = int(np.clip(y, text_h + 8, max(text_h + 8, height - baseline - 5)))
    top_left = (x - 4, y - text_h - 6)
    bottom_right = (x + text_w + 5, y + baseline + 4)
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (4, 18, 31), -1)
    cv2.addWeighted(overlay, 0.76, image, 0.24, 0, image)
    cv2.rectangle(image, top_left, bottom_right, color, 1)
    cv2.putText(
        image, text, (x, y), font, font_scale,
        (245, 250, 255), thickness, cv2.LINE_AA
    )


def draw_lower_limb_tracking(
    image_rgb,
    points,
    angles,
    visibilities,
    frame_number,
    time_seconds,
    min_visibility=0.50,
    show_angle_labels=True,
):
    """
    Draw frame-by-frame sagittal lower-limb tracking.

    Left side: cyan. Right side: violet.
    Main tracked joints: hip, knee, ankle, plus shoulder/heel/foot context.
    This is a 2D markerless visualization, not a 3D laboratory marker model.
    """
    canvas = image_rgb.copy()
    left_color = (34, 211, 238)
    right_color = (192, 132, 252)
    neutral_color = (226, 232, 240)
    muted_color = (148, 163, 184)

    side_config = {
        'Left': {
            'prefix': 'L', 'color': left_color,
            'chain': ['shoulder', 'hip', 'knee', 'ankle', 'foot'],
        },
        'Right': {
            'prefix': 'R', 'color': right_color,
            'chain': ['shoulder', 'hip', 'knee', 'ankle', 'foot'],
        },
    }

    # Header
    header = canvas.copy()
    cv2.rectangle(header, (0, 0), (canvas.shape[1], 48), (3, 16, 28), -1)
    cv2.addWeighted(header, 0.78, canvas, 0.22, 0, canvas)
    cv2.putText(
        canvas,
        f'MEDICAL GAIT AI  |  FRAME {frame_number}  |  {time_seconds:.2f} s',
        (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        neutral_color, 1, cv2.LINE_AA,
    )

    for side, config in side_config.items():
        color = config['color']
        prefix = config['prefix']
        chain = config['chain']

        for first_name, second_name in zip(chain[:-1], chain[1:]):
            key_a = f'{side}_{first_name}'
            key_b = f'{side}_{second_name}'
            if (
                visibilities.get(key_a, 0.0) < min_visibility
                or visibilities.get(key_b, 0.0) < min_visibility
            ):
                continue
            cv2.line(
                canvas,
                normalized_to_pixel(points[key_a], canvas.shape),
                normalized_to_pixel(points[key_b], canvas.shape),
                color, 4, cv2.LINE_AA,
            )

        ankle_key = f'{side}_ankle'
        heel_key = f'{side}_heel'
        if (
            visibilities.get(ankle_key, 0.0) >= min_visibility
            and visibilities.get(heel_key, 0.0) >= min_visibility
        ):
            cv2.line(
                canvas,
                normalized_to_pixel(points[ankle_key], canvas.shape),
                normalized_to_pixel(points[heel_key], canvas.shape),
                color, 3, cv2.LINE_AA,
            )

        for joint_name in ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'foot']:
            key = f'{side}_{joint_name}'
            if key not in points or visibilities.get(key, 0.0) < min_visibility:
                continue
            center = normalized_to_pixel(points[key], canvas.shape)
            radius = 8 if joint_name in ['hip', 'knee', 'ankle'] else 5
            cv2.circle(canvas, center, radius + 3, (5, 24, 39), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, neutral_color, 1, cv2.LINE_AA)

        if show_angle_labels:
            labels = [
                ('hip', f'{prefix} HIP {angles[f"{side}_hip"]:+.1f} deg'),
                ('knee', f'{prefix} KNEE {angles[f"{side}_knee"]:.1f} deg'),
                ('ankle', f'{prefix} ANKLE {angles[f"{side}_ankle"]:+.1f} deg'),
            ]
            for joint_name, label in labels:
                key = f'{side}_{joint_name}'
                if visibilities.get(key, 0.0) < min_visibility:
                    continue
                joint_px = normalized_to_pixel(points[key], canvas.shape)
                x_offset = 12 if side == 'Left' else -125
                draw_text_box(
                    canvas, label,
                    (joint_px[0] + x_offset, joint_px[1] - 10),
                    color,
                )

    if (
        visibilities.get('Left_hip', 0.0) >= min_visibility
        and visibilities.get('Right_hip', 0.0) >= min_visibility
    ):
        cv2.line(
            canvas,
            normalized_to_pixel(points['Left_hip'], canvas.shape),
            normalized_to_pixel(points['Right_hip'], canvas.shape),
            muted_color, 2, cv2.LINE_AA,
        )

    # Legend
    legend_y = canvas.shape[0] - 18
    cv2.circle(canvas, (18, legend_y - 5), 5, left_color, -1)
    cv2.putText(canvas, 'LEFT', (30, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, neutral_color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (88, legend_y - 5), 5, right_color, -1)
    cv2.putText(canvas, 'RIGHT', (100, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, neutral_color, 1, cv2.LINE_AA)
    return canvas


def draw_pose_not_detected(image_rgb, frame_number, time_seconds):
    """Overlay status when no pose is detected in the current frame."""
    canvas = image_rgb.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 48), (3, 16, 28), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.putText(
        canvas,
        f'FRAME {frame_number}  |  {time_seconds:.2f} s  |  POSE NOT DETECTED',
        (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (251, 113, 133), 1, cv2.LINE_AA,
    )
    return canvas


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
    แปลง Mean Absolute Error (°) เป็นคะแนน 0-100
    ด้วย smooth inverse-square decay:

        score = 100 / (1 + (error / tolerance)^2)

    คุณสมบัติ:
    - error = 0        -> 100 คะแนน
    - error = tolerance -> 50 คะแนน
    - ไม่มีการตัดเป็น 0 ทันทีเมื่อ error เกิน tolerance

    tolerance_deg เป็น system scaling constant
    ไม่ใช่ clinical diagnostic cut-off
    """

    if not np.isfinite(mae_deg):
        return np.nan

    if tolerance_deg <= 0:
        return 0.0

    ratio = (
        mae_deg
        / tolerance_deg
    )

    score = (
        100.0
        / (
            1.0
            + ratio ** 2
        )
    )

    return float(
        np.clip(
            score,
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
    ด้วย smooth inverse-square decay:

        score = 100 / (1 + (SI / tolerance)^2)

    0% SI หมายถึง ROM ซ้าย-ขวาเท่ากัน
    tolerance_percent เป็น system scaling constant
    ไม่ใช่ clinical diagnostic cut-off
    """

    if not np.isfinite(rom_si):
        return np.nan

    if tolerance_percent <= 0:
        return 0.0

    ratio = (
        rom_si
        / tolerance_percent
    )

    score = (
        100.0
        / (
            1.0
            + ratio ** 2
        )
    )

    return float(
        np.clip(
            score,
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
    ตรวจ estimated stride anchor แบบ heuristic จากตำแหน่งส้นเท้า
    ที่อยู่ด้านหน้าสุดเมื่อเทียบกับ pelvis

    ใช้เป็นตัวประมาณสำหรับ normalize stride cycle ในวิดีโอด้านข้าง
    ไม่ใช่ ground-contact / heel-strike ที่ยืนยันด้วย force plate

    เหมาะกับ:
    - วิดีโอด้านข้าง
    - เห็นเท้าและส้นเท้าชัด
    - กล้องนิ่ง
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
    สร้าง stride cycle แยกซ้าย/ขวาจาก estimated stride anchors
    แล้ว normalize เป็น 0-100%

    event ในเวอร์ชันนี้เป็น heuristic จากวิดีโอ 2D
    ไม่ใช่ force-plate ground-contact event
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
    # ไม่ใช่เกณฑ์วินิจฉัยทางคลินิก

    if score >= 80:

        level = "normal"

        status = (
            "ความสมมาตรโดยรวมอยู่ในระดับดี"
        )

        description = (
            "ตัวชี้วัดที่ระบบตรวจได้มีความใกล้เคียงกันโดยรวม "
            "ควรอ่านร่วมกับ joint-angle curves, ROM และ phase metrics"
        )

    elif score >= 65:

        level = "mild"

        status = (
            "พบความแตกต่างเล็กน้อย"
        )

        description = (
            "ระบบพบความแตกต่างซ้าย–ขวาเล็กน้อยในบางองค์ประกอบ "
            "ควรพิจารณาร่วมกับกราฟ gait cycle และ ROM"
        )

    elif score >= 45:

        level = "warning"

        status = (
            "ควรประเมินเพิ่มเติม"
        )

        description = (
            "ระบบพบความแตกต่างของการเคลื่อนไหวบางองค์ประกอบ "
            "ควรตรวจ joint-angle curve, ROM, gait phase "
            "และคุณภาพวิดีโอเพิ่มเติม"
        )

    else:

        level = "danger"

        status = (
            "พบความแตกต่างซ้าย–ขวาค่อนข้างมาก"
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

    st.markdown("### Live Joint Tracking")

    show_live_tracking = st.checkbox(
        "แสดงภาพการจับข้อต่อทุกเฟรม",
        value=True,
        help=(
            "อัปเดตภาพ Hip / Knee / Ankle ในทุกเฟรมที่ประมวลผล "
            "การเปิดใช้งานอาจทำให้วิดีโอยาวประมวลผลช้าลงเล็กน้อย"
        )
    )

    show_angle_labels = st.checkbox(
        "แสดงค่ามุมบนภาพ",
        value=True
    )

    min_landmark_visibility = st.slider(
        "Landmark visibility ขั้นต่ำ",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="จุดที่มี visibility ต่ำกว่าค่านี้จะไม่ถูกวาดบนภาพ"
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
    '<div class="section-title">Video Input · Sagittal View</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'ระบบจะพยายามตรวจ estimated stride anchors และ normalize การเดินเป็น 0–100% stride cycle '
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

    uploaded_suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if uploaded_suffix not in {
        ".mp4",
        ".mov",
        ".avi"
    }:
        uploaded_suffix = ".mp4"

    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=uploaded_suffix
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

    if not cap.isOpened():

        try:
            os.remove(video_path)
        except OSError:
            pass

        st.error(
            "ไม่สามารถเปิดไฟล์วิดีโอด้วย OpenCV ได้ "
            "กรุณาลอง MP4 (H.264), MOV หรือ AVI ที่อ่านได้ตามปกติ"
        )
        st.stop()

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
            "Video Frames",
            f"{total_frames:,}"
        )

    with info2:

        st.metric(
            "Frame Rate",
            f"{fps:.1f} FPS"
        )

    with info3:

        st.metric(
            "Duration",
            f"{duration:.1f} s"
        )


    # =====================================================
    # เตรียมตัวแปร
    # =====================================================

    frames_data = []

    last_pose_image = None

    frame_count = 0

    pose_detected_count = 0

    st.markdown(
        '<div class="section-title">Live Joint Tracking · Frame-by-frame</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "ติดตาม Hip / Knee / Ankle ทุกเฟรม: cyan = Left, violet = Right. "
        "ค่ามุมเป็นผลจากการวิเคราะห์ 2D sagittal-plane ของระบบ"
    )

    live_frame_col, live_status_col = st.columns(
        [1.75, 0.65],
        gap="medium"
    )

    with live_frame_col:
        st_frame = st.empty()

    with live_status_col:
        live_status = st.empty()

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

                tracking_points = {
                    "Left_shoulder": l_shoulder,
                    "Left_hip": l_hip,
                    "Left_knee": l_knee,
                    "Left_ankle": l_ankle,
                    "Left_heel": l_heel,
                    "Left_foot": l_foot,
                    "Right_shoulder": r_shoulder,
                    "Right_hip": r_hip,
                    "Right_knee": r_knee,
                    "Right_ankle": r_ankle,
                    "Right_heel": r_heel,
                    "Right_foot": r_foot,
                }

                tracking_visibilities = {
                    "Left_shoulder": float(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility),
                    "Left_hip": float(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility),
                    "Left_knee": float(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility),
                    "Left_ankle": float(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility),
                    "Left_heel": float(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].visibility),
                    "Left_foot": float(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility),
                    "Right_shoulder": float(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility),
                    "Right_hip": float(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility),
                    "Right_knee": float(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility),
                    "Right_ankle": float(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility),
                    "Right_heel": float(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].visibility),
                    "Right_foot": float(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility),
                }


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
                        right_heel_forward,

                    "Left Hip Visibility": tracking_visibilities["Left_hip"],
                    "Left Knee Visibility": tracking_visibilities["Left_knee"],
                    "Left Ankle Visibility": tracking_visibilities["Left_ankle"],
                    "Right Hip Visibility": tracking_visibilities["Right_hip"],
                    "Right Knee Visibility": tracking_visibilities["Right_knee"],
                    "Right Ankle Visibility": tracking_visibilities["Right_ankle"]
                })

                pose_detected_count += 1

                joint_angles = {
                    "Left_hip": left_hip_angle,
                    "Left_knee": left_knee_angle,
                    "Left_ankle": left_ankle_angle,
                    "Right_hip": right_hip_angle,
                    "Right_knee": right_knee_angle,
                    "Right_ankle": right_ankle_angle,
                }

                display_image = draw_lower_limb_tracking(
                    image,
                    tracking_points,
                    joint_angles,
                    tracking_visibilities,
                    frame_number=frame_count,
                    time_seconds=(frame_count / fps),
                    min_visibility=min_landmark_visibility,
                    show_angle_labels=show_angle_labels
                )

                last_pose_image = display_image.copy()

            else:

                display_image = draw_pose_not_detected(
                    image,
                    frame_number=frame_count,
                    time_seconds=(frame_count / fps)
                )


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

            detection_rate_live = (
                pose_detected_count
                / max(frame_count, 1)
                * 100.0
            )

            status_text.text(
                f"กำลังวิเคราะห์เฟรม "
                f"{frame_count:,}/{total_frames:,}"
            )

            # แสดงภาพการจับข้อต่อแบบ frame-by-frame
            if show_live_tracking:
                st_frame.image(
                    display_image,
                    channels="RGB",
                    use_container_width=True
                )

            live_status_html = (
                '<div class="med-card">'
                '<div class="card-title">Frame Telemetry</div>'
                f'<div class="small-muted">'
                f'Frame <b>{frame_count:,}</b><br>'
                f'Time <b>{frame_count / fps:.2f} s</b><br>'
                f'Pose detected <b>{pose_detected_count:,}</b><br>'
                f'Detection rate <b>{detection_rate_live:.1f}%</b>'
                '</div>'
                '</div>'
            )

            live_status.markdown(
                live_status_html,
                unsafe_allow_html=True
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

        final_detection_rate = (
            pose_detected_count
            / max(frame_count, 1)
            * 100.0
        )

        st.success(
            f"วิเคราะห์เสร็จสิ้น · "
            f"ตรวจพบ pose {pose_detected_count:,}/{frame_count:,} เฟรม "
            f"({final_detection_rate:.1f}%)"
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
            '<div class="section-title">Clinical Gait Dashboard</div>',
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
                        เฟรมล่าสุดพร้อมการติดตาม Hip / Knee / Ankle และค่ามุมซ้าย–ขวา
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
        # CENTER: Gait Screening Score + component chart
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

            status_css_class = (
                f"score-status-{screening['level']}"
            )

            # สร้าง HTML โดยไม่เยื้องภายใน string
            # เพื่อป้องกัน Streamlit Markdown ตีความเป็น code block
            score_html = (
                f'<div class="med-card">\n'
                f'<div class="card-title">Gait Screening Score</div>\n'
                f'<div class="score-number">'
                f'{screening["score"]:.0f}'
                f'<span class="score-unit">/100</span>'
                f'</div>\n'
                f'<div class="screening-status {status_css_class}">'
                f'{screening["status"]}'
                f'</div>\n'
                f'<div class="score-caption">'
                f'Curve MAE {curve_mae_caption} &nbsp;·&nbsp; '
                f'ROM SI {screening["overall_rom_si"]:.2f}% &nbsp;·&nbsp; '
                f'Phase Difference {phase_mae_caption}'
                f'</div>\n'
                f'</div>\n'
                f'<div class="model-strip">\n'
                f'<div class="model-cell">'
                f'<div class="model-weight">40%</div>'
                f'<div class="model-label">Joint Curve</div>'
                f'</div>\n'
                f'<div class="model-cell">'
                f'<div class="model-weight">35%</div>'
                f'<div class="model-label">ROM Symmetry</div>'
                f'</div>\n'
                f'<div class="model-cell">'
                f'<div class="model-weight">25%</div>'
                f'<div class="model-label">Phase / Peak</div>'
                f'</div>\n'
                f'</div>'
            )

            st.markdown(
                score_html,
                unsafe_allow_html=True
            )

            # Gauge ต้องอยู่ใน with center_col ระดับเดียวกับ st.markdown
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=screening["score"],
                    number={
                        "font": {"size": 48}
                    },
                    title={
                        "text": "Prototype score / 100",
                        "font": {"size": 14}
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
                                "range": [0, 45],
                                "color": "rgba(251,113,133,0.34)"
                            },
                            {
                                "range": [45, 65],
                                "color": "rgba(251,146,60,0.34)"
                            },
                            {
                                "range": [65, 80],
                                "color": "rgba(250,204,21,0.30)"
                            },
                            {
                                "range": [80, 100],
                                "color": "rgba(52,211,153,0.32)"
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
                    message = "ความสมมาตรอยู่ในระดับดี"

                elif joint_score >= 65:

                    css_class = "status-mild"
                    message = "พบความแตกต่างเล็กน้อย"

                elif joint_score >= 45:

                    css_class = "status-watch"
                    message = "ควรประเมินเพิ่มเติม"

                else:

                    css_class = "status-alert"
                    message = "พบความแตกต่างค่อนข้างมาก"

                st.markdown(
                    f'<div class="{css_class}">'
                    f'<b>{label}</b><br>'
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
                    <b>ข้อควรระวัง</b><br>
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
            '<div class="section-title">Scoring & Quality Metrics</div>',
            unsafe_allow_html=True
        )

        score_c1, score_c2, score_c3 = st.columns(3)

        with score_c1:

            if np.isfinite(
                screening["curve_component_score"]
            ):

                st.metric(
                    "Curve Similarity Score",
                    f"{screening['curve_component_score']:.0f} / 100",
                    help="น้ำหนัก 40% ของคะแนนรวม"
                )

            else:

                st.metric(
                    "Curve Similarity Score",
                    "N/A",
                    help="ต้องมี normalized gait cycle"
                )

        with score_c2:

            st.metric(
                "ROM Symmetry Score",
                f"{screening['rom_component_score']:.0f} / 100",
                help="น้ำหนัก 35% ของคะแนนรวม"
            )

        with score_c3:

            if np.isfinite(
                screening["phase_component_score"]
            ):

                st.metric(
                    "Phase / Peak Score",
                    f"{screening['phase_component_score']:.0f} / 100",
                    help="น้ำหนัก 25% ของคะแนนรวม"
                )

            else:

                st.metric(
                    "Phase / Peak Score",
                    "N/A",
                    help="ต้องมี normalized gait cycle"
                )

        st.caption(
            "สูตรคะแนนใช้ smooth decay เพื่อหลีกเลี่ยงการตัดคะแนนเป็นศูนย์ทันที "
            "เมื่อค่าความต่างเกิน tolerance; tolerance เป็นค่าปรับสเกลของ prototype "
            "ไม่ใช่ clinical diagnostic cut-off"
        )

        m1, m2, m3, m4, m5 = st.columns(5)

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
                "Valid Pose Frames",
                f"{len(df):,}"
            )

        with m4:
            st.metric(
                "Analyzed Duration",
                f"{df['Time (s)'].max():.1f} s"
            )

        with m5:
            st.metric(
                "Pose Detection",
                f"{final_detection_rate:.1f}%"
            )


        # =================================================
        # Gait Cycle 0-100%
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">Normalized Gait Cycle · 0–100%</div>',
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
                'Phase-based Reference Comparison'
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
                "ยังตรวจพบ estimated stride anchors ต่อเนื่องไม่เพียงพอ "
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
            '<div class="section-title">Joint Angle Time Series</div>',
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
            '<div class="section-title">Knee Kinematics Summary</div>',
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
            '<div class="section-title">Left–Right Joint Difference</div>',
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
            '<div class="section-title">Joint-Level Detail</div>',
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
            '<div class="section-title">Joint Angle Summary Table</div>',
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
            '<div class="section-title">Interpretation Framework</div>',
            unsafe_allow_html=True
        )


        interpretation_df = pd.DataFrame({

            "ระดับระบบ": [
                "ระดับดี",
                "แตกต่างเล็กน้อย",
                "ควรประเมินเพิ่มเติม",
                "แตกต่างค่อนข้างมาก"
            ],

            "Gait Screening Score": [
                "80–100",
                "65–79",
                "45–64",
                "0–44"
            ],

            "ความหมาย": [
                (
                    "ตัวชี้วัดซ้าย–ขวามีความใกล้เคียงกันโดยรวม "
                    "ตามโมเดลคะแนนของระบบ"
                ),
                (
                    "พบความแตกต่างเล็กน้อยในบางองค์ประกอบ"
                ),
                (
                    "พบความแตกต่างของการเคลื่อนไหวบางองค์ประกอบ "
                    "ควรตรวจกราฟและคุณภาพข้อมูลเพิ่มเติม"
                ),
                (
                    "พบความแตกต่างหลายองค์ประกอบ "
                    "ควรตรวจคุณภาพวิดีโอและประเมินเพิ่มเติม"
                )
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

            Gait Screening Score และเกณฑ์สีในระบบนี้เป็น
            **prototype screening model** ที่รวม 3 องค์ประกอบ:
            Joint Curve Similarity 40%, ROM Symmetry 35% และ
            Phase / Peak Similarity 25%

            การแปลงค่าความต่างเป็นคะแนนใช้ smooth decay
            `100 / (1 + (error / tolerance)^2)` เพื่อลดปัญหา
            คะแนนตกเป็นศูนย์ทันทีเมื่อค่าความต่างเกิน tolerance

            ค่า tolerance และช่วงคะแนนสีเป็นค่าปรับสเกลของระบบ
            ไม่ใช่ clinical diagnostic cut-off และยังไม่ได้ผ่าน
            clinical validation

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
            '<div class="section-title">Data Export</div>',
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
            '<div class="section-title">Assessment Context</div>',
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
