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
    page_title="ระบบวิเคราะห์ท่าเดินจากวิดีโอ",
    page_icon="🦶",
    layout="wide"
)
st.title("🦶 ระบบวิเคราะห์ท่าเดินจากวิดีโอ")
st.caption(
    "Video Gait Analysis System | "
    "วิเคราะห์มุมข้อต่อและดัชนีความสมมาตรจากวิดีโอ"
)
# =========================================================
# 3. ฟังก์ชันคำนวณมุมข้อต่อ
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
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)
# =========================================================
# 4. ฟังก์ชันคำนวณ Symmetry Index
# =========================================================
def calculate_symmetry_index(left_val, right_val):
    if left_val + right_val == 0:
        return 0.0
    return (
        abs(left_val - right_val)
        / (0.5 * (left_val + right_val))
    ) * 100
# =========================================================
# 5. ฟังก์ชันคำนวณ ROM
# =========================================================
def calculate_rom(series):
    return float(series.max() - series.min()) 
    # =========================================================
# 6. ฟังก์ชันประเมิน Gait Screening
# =========================================================
def calculate_gait_screening(df):
    """
    ประเมินความสมมาตรของการเคลื่อนไหวเบื้องต้น
    ใช้สำหรับ Gait Screening เท่านั้น
    ไม่ใช่การวินิจฉัยทางการแพทย์
    """

    # -----------------------------------------------------
    # คำนวณค่าเฉลี่ยของมุมแต่ละข้าง
    # -----------------------------------------------------
    left_knee = df["Left Knee Angle"].mean()
    right_knee = df["Right Knee Angle"].mean()

    left_hip = df["Left Hip Angle"].mean()
    right_hip = df["Right Hip Angle"].mean()

    left_ankle = df["Left Ankle Angle"].mean()
    right_ankle = df["Right Ankle Angle"].mean()

    # -----------------------------------------------------
    # คำนวณ Symmetry Index
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # ค่าเฉลี่ย Symmetry Index รวม
    # -----------------------------------------------------
    overall_si = np.mean([
        knee_si,
        hip_si,
        ankle_si
    ])

    # -----------------------------------------------------
    # แปลงเป็นคะแนน Gait Screening Score
    #
    # SI ต่ำ = สมมาตรมาก = คะแนนสูง
    # -----------------------------------------------------
    score = max(
        0,
        min(
            100,
            100 - (overall_si * 2)
        )
    )

    # -----------------------------------------------------
    # ประเมินระดับ
    # -----------------------------------------------------
    if overall_si < 5:
        status = "🟢 ปกติ"
        description = (
            "พบความแตกต่างระหว่างซ้ายและขวาในระดับต่ำ "
            "จากข้อมูลมุมข้อต่อที่วิเคราะห์ได้"
        )

    elif overall_si < 10:
        status = "🟡 ควรประเมินเพิ่มเติม"
        description = (
            "พบความแตกต่างระหว่างซ้ายและขวาในระดับปานกลาง "
            "ควรพิจารณาข้อมูลเพิ่มเติมและสังเกตลักษณะการเดิน"
        )

    else:
        status = "🔴 พบความแตกต่างมาก"
        description = (
            "พบความแตกต่างระหว่างซ้ายและขวาค่อนข้างมาก "
            "ควรได้รับการประเมินการเดินเพิ่มเติมโดยผู้เชี่ยวชาญ"
        )

    return {
        "status": status,
        "score": score,
        "overall_si": overall_si,
        "knee_si": knee_si,
        "hip_si": hip_si,
        "ankle_si": ankle_si,
        "description": description
    }
# =========================================================
# 6. อัปโหลดวิดีโอ
# =========================================================
uploaded_file = st.file_uploader(
    "📹 อัปโหลดคลิปวิดีโอการเดิน",
    type=["mp4", "mov", "avi"]
)
# =========================================================
# 7. เริ่มวิเคราะห์เมื่อมีวิดีโอ
# =========================================================
if uploaded_file is not None:
    # -----------------------------------------------------
    # บันทึกไฟล์วิดีโอชั่วคราว
    # -----------------------------------------------------
    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name
    # -----------------------------------------------------
    # เปิดวิดีโอ
    # -----------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ) or 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    st.info(
        f"อัปโหลดวิดีโอเรียบร้อยแล้ว | "
        f"ประมาณ {total_frames} เฟรม | "
        f"{fps:.1f} FPS"
    )
    # -----------------------------------------------------
    # เตรียมตัวแปร
    # -----------------------------------------------------
    frames_data = []
    frame_count = 0
    st_frame = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    # =====================================================
    # 8. MediaPipe Pose
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
            # แปลง BGR → RGB
            # -------------------------------------------------
            image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            image.flags.writeable = False
            # -------------------------------------------------
            # วิเคราะห์ Pose
            # -------------------------------------------------
            results = pose.process(image)
            image.flags.writeable = True
            # =================================================
            # 9. ถ้าพบโครงร่างร่างกาย
            # =================================================
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
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
                # 10. คำนวณมุมข้อต่อ
                # =================================================
                # Knee
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
                # Hip
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
                # Ankle
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
                # 11. เก็บข้อมูลของแต่ละ Frame
                # =================================================
                frames_data.append({
                    "Frame": frame_count,
                    "Time (s)": frame_count / fps,
                    "Left Hip Angle": left_hip_angle,
                    "Right Hip Angle": right_hip_angle,
                    "Left Knee Angle": left_knee_angle,
                    "Right Knee Angle": right_knee_angle,
                    "Left Ankle Angle": left_ankle_angle,
                    "Right Ankle Angle": right_ankle_angle
                })
                # =================================================
                # 12. วาด Skeleton
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
            # 13. แสดง Progress
            # =================================================
            progress = min(
                frame_count / total_frames,
                1.0
            )
            progress_bar.progress(progress)
            status_text.text(
                f"กำลังประมวลผลเฟรมที่ "
                f"{frame_count}/{total_frames}"
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
    # =========================================================
    # 14. ปิด Video
    # =========================================================
    cap.release()
    try:
        os.remove(video_path)
    except Exception:
        pass
    progress_bar.empty()
    status_text.empty()
    st.success("✅ วิเคราะห์วิดีโอเสร็จสิ้น!")
    # =========================================================
    # 15. วิเคราะห์ข้อมูล
    # =========================================================
    if frames_data:
        df = pd.DataFrame(frames_data)
        # =====================================================
        # 16. แสดงข้อมูลเบื้องต้น
        # =====================================================
        st.subheader("📋 ข้อมูลการวิเคราะห์")
        st.dataframe(
            df.head(10),
            use_container_width=True
        )
        # =====================================================
        # 17. กราฟ Knee
        # =====================================================
        st.subheader(
            "🦵 การเปลี่ยนแปลงมุมเข่า"
        )
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
            title="Left vs Right Knee Angle"
        )
        st.plotly_chart(
            fig_knee,
            use_container_width=True
        )
        # =====================================================
        # 18. กราฟ Hip
        # =====================================================
        st.subheader(
            "🦿 การเปลี่ยนแปลงมุมสะโพก"
        )
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
            title="Left vs Right Hip Angle"
        )
        st.plotly_chart(
            fig_hip,
            use_container_width=True
        )
        # =====================================================
        # 19. กราฟ Ankle
        # =====================================================
        st.subheader(
            "🦶 การเปลี่ยนแปลงมุมข้อเท้า"
        )
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
            title="Left vs Right Ankle Angle"
        )
        st.plotly_chart(
            fig_ankle,
            use_container_width=True
        )
        # =====================================================
        # 20. คำนวณสถิติ
        # =====================================================
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
        # =====================================================
        # 21. Symmetry Index
        # =====================================================
        si_mean_knee = calculate_symmetry_index(
            mean_left_knee,
            mean_right_knee
        )
        si_max_knee = calculate_symmetry_index(
            max_left_knee,
            max_right_knee
        )
        # =====================================================
        # 22. Dashboard สรุป
        # =====================================================
        st.subheader(
            "📈 สรุปผลการวิเคราะห์มุมเข่า"
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "เข่าซ้ายเฉลี่ย",
                f"{mean_left_knee:.2f}°"
            )
            st.metric(
                "เข่าซ้ายสูงสุด",
                f"{max_left_knee:.2f}°"
            )
        with col2:
            st.metric(
                "เข่าขวาเฉลี่ย",
                f"{mean_right_knee:.2f}°"
            )
            st.metric(
                "เข่าขวาสูงสุด",
                f"{max_right_knee:.2f}°"
            )
        with col3:
            st.metric(
                "ROM ซ้าย",
                f"{rom_left_knee:.2f}°"
            )
            st.metric(
                "ROM ขวา",
                f"{rom_right_knee:.2f}°"
            )
        with col4:
            st.metric(
                "Symmetry Index",
                f"{si_mean_knee:.2f}%"
            )
            st.metric(
                "SI สูงสุด",
                f"{si_max_knee:.2f}%"
            )
        # =====================================================
        # 23. ตารางสรุป
        # =====================================================
        st.subheader(
            "📊 ตารางสรุปมุมข้อต่อ"
        )
        summary_data = {
            "ข้อต่อ": [
                "Hip",
                "Knee",
                "Ankle"
            ],
            "ซ้ายเฉลี่ย (°)": [
                df["Left Hip Angle"].mean(),
                df["Left Knee Angle"].mean(),
                df["Left Ankle Angle"].mean()
            ],
            "ขวาเฉลี่ย (°)": [
                df["Right Hip Angle"].mean(),
                df["Right Knee Angle"].mean(),
                df["Right Ankle Angle"].mean()
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
            summary_df,
            use_container_width=True
        )
        # =====================================================
        # 24. ดาวน์โหลด CSV
        # =====================================================
        st.divider()
        csv_data = df.to_csv(
            index=False
        ).encode("utf-8-sig")
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูลทั้งหมด (.CSV)",
            data=csv_data,
            file_name="gait_analysis_data.csv",
            mime="text/csv"
        )
