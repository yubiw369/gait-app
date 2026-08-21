import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
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
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. CSS สำหรับ UI
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #17324d;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #17324d;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .screening-card {
        padding: 1.5rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    .score-number {
        font-size: 3rem;
        font-weight: 800;
        color: #17324d;
        text-align: center;
    }

    .score-label {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
    }

    .normal-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: #ecfdf5;
        border: 1px solid #86efac;
    }

    .warning-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: #fffbeb;
        border: 1px solid #fcd34d;
    }

    .danger-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: #fef2f2;
        border: 1px solid #fca5a5;
    }

    .info-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: #eff6ff;
        border: 1px solid #93c5fd;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. Header
# =========================================================

st.markdown(
    '<div class="main-title">🦶 ระบบวิเคราะห์ท่าเดินจากวิดีโอ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Video Gait Analysis System | '
    'วิเคราะห์มุมข้อต่อ ความสมมาตร และ Gait Screening'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 5. ฟังก์ชันคำนวณมุมข้อต่อ
# =========================================================

def calculate_angle(a, b, c):
    """
    คำนวณมุม ABC จากจุด A, B และ C
    โดยใช้พิกัด 2D (x, y)
    """

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    radians = (
        np.arctan2(
            c[1] - b[1],
            c[0] - b[0]
        )
        -
        np.arctan2(
            a[1] - b[1],
            a[0] - b[0]
        )
    )

    angle = np.abs(
        radians * 180.0 / np.pi
    )

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


# =========================================================
# 6. ฟังก์ชันคำนวณ Symmetry Index
# =========================================================

def calculate_symmetry_index(left_val, right_val):

    denominator = 0.5 * (
        abs(left_val) + abs(right_val)
    )

    if denominator == 0:
        return 0.0

    return (
        abs(left_val - right_val)
        / denominator
    ) * 100


# =========================================================
# 7. ฟังก์ชันคำนวณ ROM
# =========================================================

def calculate_rom(series):

    if series.empty:
        return 0.0

    return float(
        series.max() - series.min()
    )


# =========================================================
# 8. ฟังก์ชันประเมิน Gait Screening
# =========================================================

def calculate_gait_screening(df):
    """
    ประเมินความสมมาตรของการเคลื่อนไหวเบื้องต้น

    ใช้สำหรับ Gait Screening เท่านั้น
    ไม่ใช่การวินิจฉัยทางการแพทย์
    """

    # -----------------------------------------------------
    # ตรวจสอบข้อมูล
    # -----------------------------------------------------

    required_columns = [
        "Left Knee Angle",
        "Right Knee Angle",
        "Left Hip Angle",
        "Right Hip Angle",
        "Left Ankle Angle",
        "Right Ankle Angle"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"ไม่พบข้อมูล {column}"
            )

    if df.empty:
        raise ValueError(
            "ไม่มีข้อมูลสำหรับการประเมิน"
        )


    # =====================================================
    # ค่าเฉลี่ยมุมแต่ละข้าง
    # =====================================================

    left_knee = df[
        "Left Knee Angle"
    ].mean()

    right_knee = df[
        "Right Knee Angle"
    ].mean()

    left_hip = df[
        "Left Hip Angle"
    ].mean()

    right_hip = df[
        "Right Hip Angle"
    ].mean()

    left_ankle = df[
        "Left Ankle Angle"
    ].mean()

    right_ankle = df[
        "Right Ankle Angle"
    ].mean()


    # =====================================================
    # Symmetry Index
    # =====================================================

    knee_si = calculate_symmetry_index(
        left_knee,
        right_knee
    )

    hip_si = calculate_symmetry_index(
        left_hip,
        right_hip
    )

    ankle_si = calculate_symmetry_index(
        left_ankle,
        right_ankle
    )


    # =====================================================
    # Overall SI
    # =====================================================

    overall_si = float(
        np.mean([
            knee_si,
            hip_si,
            ankle_si
        ])
    )


    # =====================================================
    # ROM
    # =====================================================

    left_knee_rom = calculate_rom(
        df["Left Knee Angle"]
    )

    right_knee_rom = calculate_rom(
        df["Right Knee Angle"]
    )

    left_hip_rom = calculate_rom(
        df["Left Hip Angle"]
    )

    right_hip_rom = calculate_rom(
        df["Right Hip Angle"]
    )

    left_ankle_rom = calculate_rom(
        df["Left Ankle Angle"]
    )

    right_ankle_rom = calculate_rom(
        df["Right Ankle Angle"]
    )


    # =====================================================
    # ROM Symmetry Index
    # =====================================================

    knee_rom_si = calculate_symmetry_index(
        left_knee_rom,
        right_knee_rom
    )

    hip_rom_si = calculate_symmetry_index(
        left_hip_rom,
        right_hip_rom
    )

    ankle_rom_si = calculate_symmetry_index(
        left_ankle_rom,
        right_ankle_rom
    )


    # =====================================================
    # Overall ROM SI
    # =====================================================

    overall_rom_si = float(
        np.mean([
            knee_rom_si,
            hip_rom_si,
            ankle_rom_si
        ])
    )


    # =====================================================
    # Gait Screening Score
    #
    # SI ต่ำ = สมมาตรมาก
    # SI สูง = แตกต่างมาก
    #
    # หมายเหตุ:
    # เป็นคะแนน screening ที่ออกแบบในระบบ
    # ไม่ใช่คะแนนมาตรฐานทางคลินิก
    # =====================================================

    score = max(
        0.0,
        min(
            100.0,
            100.0 - (overall_si * 2.0)
        )
    )


    # =====================================================
    # ประเมินระดับ
    # =====================================================

    if overall_si < 5:

        status = "🟢 ปกติ"

        description = (
            "จากตัวชี้วัดที่ระบบวิเคราะห์ "
            "พบความแตกต่างระหว่างด้านซ้ายและขวา "
            "ในระดับต่ำ และมีความสมมาตรโดยรวมค่อนข้างดี"
        )

        recommendation = (
            "สามารถใช้ผลนี้เป็นข้อมูลคัดกรองเบื้องต้น "
            "และควรพิจารณาร่วมกับลักษณะการเดินจริง "
            "และข้อมูลอื่นที่เกี่ยวข้อง"
        )

        level = "normal"


    elif overall_si < 10:

        status = "🟡 ควรประเมินเพิ่มเติม"

        description = (
            "พบความแตกต่างระหว่างด้านซ้ายและขวา "
            "ในระดับปานกลางจากตัวชี้วัดที่ระบบใช้"
        )

        recommendation = (
            "ควรตรวจสอบวิดีโอเพิ่มเติม เช่น "
            "รูปแบบการลงเท้า การก้าว ความยาวก้าว "
            "การเคลื่อนไหวของเข่า สะโพก และข้อเท้า "
            "รวมถึงพิจารณาคุณภาพและมุมของกล้อง"
        )

        level = "warning"


    else:

        status = "🔴 พบความแตกต่างมาก"

        description = (
            "พบความแตกต่างระหว่างด้านซ้ายและขวา "
            "ค่อนข้างมากจากตัวชี้วัดที่ใช้ในการคัดกรอง"
        )

        recommendation = (
            "ควรพิจารณาประเมินการเดินเพิ่มเติมโดยผู้เชี่ยวชาญ "
            "โดยเฉพาะหากมีอาการปวด อ่อนแรง "
            "เดินผิดปกติ หรือมีปัญหาด้านการทรงตัวร่วมด้วย"
        )

        level = "danger"


    # =====================================================
    # คืนค่าผลทั้งหมด
    # =====================================================

    return {

        "status": status,

        "level": level,

        "score": score,

        "overall_si": overall_si,

        "knee_si": knee_si,

        "hip_si": hip_si,

        "ankle_si": ankle_si,

        "overall_rom_si": overall_rom_si,

        "knee_rom_si": knee_rom_si,

        "hip_rom_si": hip_rom_si,

        "ankle_rom_si": ankle_rom_si,

        "description": description,

        "recommendation": recommendation
    }


# =========================================================
# 9. Sidebar
# =========================================================

with st.sidebar:

    st.markdown("## 🦶 Gait Analysis")

    st.markdown(
        """
        **ระบบวิเคราะห์ท่าเดินจากวิดีโอ**

        ระบบใช้ Computer Vision และ
        MediaPipe Pose เพื่อวิเคราะห์

        - Hip Angle
        - Knee Angle
        - Ankle Angle
        - Symmetry Index
        - ROM
        - Gait Screening Score
        """
    )

    st.divider()

    st.markdown("### ⚠️ ข้อควรทราบ")

    st.caption(
        "ผลลัพธ์เป็นการคัดกรองเบื้องต้น "
        "จากข้อมูลวิดีโอ ไม่ใช่การวินิจฉัยโรค"
    )


# =========================================================
# 10. อัปโหลดวิดีโอ
# =========================================================

st.markdown(
    '<div class="section-title">📹 อัปโหลดวิดีโอ</div>',
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


                # =================================================
                # คำนวณ Knee
                # =================================================

                left_knee_angle = calculate_angle(
                    l_hip,
                    l_knee,
                    l_ankle
                )

                right_knee_angle = calculate_angle(
                    r_hip,
                    r_knee,
                    r_ankle
                )


                # =================================================
                # คำนวณ Hip
                # =================================================

                left_hip_angle = calculate_angle(
                    l_shoulder,
                    l_hip,
                    l_knee
                )

                right_hip_angle = calculate_angle(
                    r_shoulder,
                    r_hip,
                    r_knee
                )


                # =================================================
                # คำนวณ Ankle
                # =================================================

                left_ankle_angle = calculate_angle(
                    l_knee,
                    l_ankle,
                    l_foot
                )

                right_ankle_angle = calculate_angle(
                    r_knee,
                    r_ankle,
                    r_foot
                )


                # =================================================
                # เก็บข้อมูล
                # =================================================

                frames_data.append({

                    "Frame": frame_count,

                    "Time (s)": (
                        frame_count / fps
                    ),

                    "Left Hip Angle":
                        left_hip_angle,

                    "Right Hip Angle":
                        right_hip_angle,

                    "Left Knee Angle":
                        left_knee_angle,

                    "Right Knee Angle":
                        right_knee_angle,

                    "Left Ankle Angle":
                        left_ankle_angle,

                    "Right Ankle Angle":
                        right_ankle_angle
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
        # ส่วนที่ 1: Screening
        # =================================================

        screening = calculate_gait_screening(
            df
        )


        st.divider()

        st.markdown(
            '<div class="section-title">🩺 ผลการคัดกรองการเดิน</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "ผลนี้เป็นการคัดกรองจากความแตกต่างของมุมข้อต่อ "
            "ซ้าย–ขวา ไม่ใช่การวินิจฉัยทางการแพทย์"
        )


        # =================================================
        # Screening Score
        # =================================================

        score_col1, score_col2 = st.columns(
            [1, 2]
        )


        with score_col1:

            st.markdown(
                f"""
                <div class="screening-card">
                    <div class="score-number">
                        {screening['score']:.1f}
                    </div>
                    <div class="score-label">
                        Gait Screening Score / 100
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            st.progress(
                int(
                    round(
                        screening["score"]
                    )
                )
            )


        with score_col2:

            if screening["level"] == "normal":

                st.markdown(
                    f"""
                    <div class="normal-card">
                        <h2>🟢 ปกติ</h2>
                        <p>
                        {screening['description']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif screening["level"] == "warning":

                st.markdown(
                    f"""
                    <div class="warning-card">
                        <h2>🟡 ควรประเมินเพิ่มเติม</h2>
                        <p>
                        {screening['description']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="danger-card">
                        <h2>🔴 พบความแตกต่างมาก</h2>
                        <p>
                        {screening['description']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # =================================================
        # Overall Metrics
        # =================================================

        st.markdown(
            '<div class="section-title">📊 ตัวชี้วัดหลัก</div>',
            unsafe_allow_html=True
        )


        m1, m2, m3, m4 = st.columns(4)


        with m1:

            st.metric(
                "Overall SI",
                f"{screening['overall_si']:.2f}%"
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
        # คำอธิบาย + คำแนะนำ
        # =================================================

        desc_col, rec_col = st.columns(2)


        with desc_col:

            st.markdown(
                f"""
                <div class="info-card">

                <h3>📌 คำอธิบายผล</h3>

                <p>
                {screening['description']}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        with rec_col:

            st.markdown(
                f"""
                <div class="warning-card">

                <h3>💡 คำแนะนำ</h3>

                <p>
                {screening['recommendation']}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # ส่วนที่ 2: กราฟ
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📈 การเคลื่อนไหวของข้อต่อ</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # Knee
        # =================================================

        fig_knee = px.line(

            df,

            x="Time (s)",

            y=[
                "Left Knee Angle",
                "Right Knee Angle"
            ],

            labels={
                "value": "มุม (องศา)",
                "variable": "ข้าง",
                "Time (s)": "เวลา (วินาที)"
            },

            title="🦵 Knee Angle"
        )


        fig_knee.update_layout(
            hovermode="x unified",
            legend_title_text="",
            height=420
        )


        st.plotly_chart(
            fig_knee,
            use_container_width=True
        )


        # =================================================
        # Hip
        # =================================================

        fig_hip = px.line(

            df,

            x="Time (s)",

            y=[
                "Left Hip Angle",
                "Right Hip Angle"
            ],

            labels={
                "value": "มุม (องศา)",
                "variable": "ข้าง",
                "Time (s)": "เวลา (วินาที)"
            },

            title="🦿 Hip Angle"
        )


        fig_hip.update_layout(
            hovermode="x unified",
            legend_title_text="",
            height=420
        )


        st.plotly_chart(
            fig_hip,
            use_container_width=True
        )


        # =================================================
        # Ankle
        # =================================================

        fig_ankle = px.line(

            df,

            x="Time (s)",

            y=[
                "Left Ankle Angle",
                "Right Ankle Angle"
            ],

            labels={
                "value": "มุม (องศา)",
                "variable": "ข้าง",
                "Time (s)": "เวลา (วินาที)"
            },

            title="🦶 Ankle Angle"
        )


        fig_ankle.update_layout(
            hovermode="x unified",
            legend_title_text="",
            height=420
        )


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
            "Left Knee Angle"
        ].mean()

        mean_right_knee = df[
            "Right Knee Angle"
        ].mean()

        max_left_knee = df[
            "Left Knee Angle"
        ].max()

        max_right_knee = df[
            "Right Knee Angle"
        ].max()

        min_left_knee = df[
            "Left Knee Angle"
        ].min()

        min_right_knee = df[
            "Right Knee Angle"
        ].min()

        rom_left_knee = calculate_rom(
            df["Left Knee Angle"]
        )

        rom_right_knee = calculate_rom(
            df["Right Knee Angle"]
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
            '<div class="section-title">🔎 Symmetry Index รายข้อต่อ</div>',
            unsafe_allow_html=True
        )


        si_df = pd.DataFrame({

            "ข้อต่อ": [
                "Hip",
                "Knee",
                "Ankle"
            ],

            "Symmetry Index (%)": [
                screening["hip_si"],
                screening["knee_si"],
                screening["ankle_si"]
            ],

            "ROM Symmetry Index (%)": [
                screening["hip_rom_si"],
                screening["knee_rom_si"],
                screening["ankle_rom_si"]
            ]
        })


        st.dataframe(
            si_df.style.format({
                "Symmetry Index (%)": "{:.2f}",
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

            st.metric(
                "Symmetry Index",
                f"{screening['hip_si']:.2f}%"
            )

            st.metric(
                "ROM SI",
                f"{screening['hip_rom_si']:.2f}%"
            )


        with c2:

            st.markdown("### 🦵 Knee")

            st.metric(
                "Symmetry Index",
                f"{screening['knee_si']:.2f}%"
            )

            st.metric(
                "ROM SI",
                f"{screening['knee_rom_si']:.2f}%"
            )


        with c3:

            st.markdown("### 🦶 Ankle")

            st.metric(
                "Symmetry Index",
                f"{screening['ankle_si']:.2f}%"
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
                    "Left Hip Angle"
                ].mean(),

                df[
                    "Left Knee Angle"
                ].mean(),

                df[
                    "Left Ankle Angle"
                ].mean()
            ],

            "ขวาเฉลี่ย (°)": [

                df[
                    "Right Hip Angle"
                ].mean(),

                df[
                    "Right Knee Angle"
                ].mean(),

                df[
                    "Right Ankle Angle"
                ].mean()
            ],

            "ซ้าย ROM (°)": [

                calculate_rom(
                    df["Left Hip Angle"]
                ),

                calculate_rom(
                    df["Left Knee Angle"]
                ),

                calculate_rom(
                    df["Left Ankle Angle"]
                )
            ],

            "ขวา ROM (°)": [

                calculate_rom(
                    df["Right Hip Angle"]
                ),

                calculate_rom(
                    df["Right Knee Angle"]
                ),

                calculate_rom(
                    df["Right Ankle Angle"]
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

            Gait Screening Score และเกณฑ์การแปลผลในระบบนี้
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
