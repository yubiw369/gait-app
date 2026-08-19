import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import tempfile
import os
import mediapipe as mp

# ==========================================
# 1. โหลด MediaPipe Solutions แบบสะอาด
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==========================================
# 2. ตั้งค่าหน้าเว็บ Streamlit
# ==========================================
st.set_page_config(
    page_title="ระบบวิเคราะห์ท่าเดินจากวิดีโอ",
    page_icon="🦶",
    layout="wide"
)

st.title("🦶 ระบบวิเคราะห์ท่าเดินจากวิดีโอ (Video Gait Analysis System)")
st.caption("พัฒนาขึ้นเพื่อการวิเคราะห์มุมข้อต่อชีวกลศาสตร์ และดัชนีความสมมาตร (Symmetry Index)")

# ==========================================
# 3. ฟังก์ชันคำนวณทางชีวกลศาสตร์
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

def calculate_symmetry_index(left_val, right_val):
    """คำนวณ Symmetry Index (SI) ยิ่งเข้าใกล้ 0% ยิ่งสมมาตรกัน"""
    if left_val + right_val == 0:
        return 0.0
    return (abs(left_val - right_val) / (0.5 * (left_val + right_val))) * 100

# ==========================================
# 4. ส่วนการอัปโหลดและประมวลผล
# ==========================================
uploaded_file = st.file_uploader("อัปโหลดคลิปวิดีโอการเดิน (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    st.info(f"อัปโหลดวิดีโอเรียบร้อยแล้ว (จำนวนทั้งหมดประมาณ {total_frames} เฟรม)")

    frames_data = []
    frame_count = 0

    st_frame = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            image.flags.writeable = True
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # พิกัดขาซ้าย
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # พิกัดขาขวา
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                
                frames_data.append({
                    'Frame': frame_count,
                    'Left Knee Angle': left_knee_angle,
                    'Right Knee Angle': right_knee_angle
                })

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # อัปเดต Progress และวิดีโอ
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"กำลังประมวลผลเฟรมที่: {frame_count}/{total_frames}")

            if frame_count % 2 == 0:
                st_frame.image(image, channels="RGB", use_container_width=True)

    cap.release()
    try:
        os.remove(video_path)
    except Exception:
        pass

    progress_bar.empty()
    status_text.empty()
    st.success("วิเคราะห์วิดีโอเสร็จสิ้น!")

    # ==========================================
    # 5. แสดงผลลัพธ์ข้อมูลและดาวน์โหลด
    # ==========================================
    if frames_data:
        df = pd.DataFrame(frames_data)
        
        # กราฟ
        st.subheader("📊 กราฟแสดงการเปลี่ยนแปลงมุมเข่า (Knee Joint Angle)")
        fig = px.line(
            df, 
            x='Frame', 
            y=['Left Knee Angle', 'Right Knee Angle'],
            labels={'value': 'มุม (องศา)', 'variable': 'ข้าง'},
            title="การเปรียบเทียบมุมเข่าซ้ายและขวาตลอดช่วงการเดิน"
        )
        st.plotly_chart(fig, use_container_width=True)

        # คำนวณค่าทางสถิติและ Symmetry Index
        mean_left = df['Left Knee Angle'].mean()
        mean_right = df['Right Knee Angle'].mean()
        si_mean = calculate_symmetry_index(mean_left, mean_right)

        max_left = df['Left Knee Angle'].max()
        max_right = df['Right Knee Angle'].max()
        si_max = calculate_symmetry_index(max_left, max_right)

        # สรุปค่าสถิติ
        st.subheader("📈 สรุปค่าสถิติและดัชนีความสมมาตร")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("มุมเข่าซ้ายเฉลี่ย", f"{mean_left:.2f}°")
            st.metric("มุมเข่าซ้ายสูงสุด", f"{max_left:.2f}°")
        with col2:
            st.metric("มุมเข่าขวาเฉลี่ย", f"{mean_right:.2f}°")
            st.metric("มุมเข่าขวาสูงสุด", f"{max_right:.2f}°")
        with col3:
            st.metric("Symmetry Index (เฉลี่ย)", f"{si_mean:.2f}%", help="ยิ่งใกล้ 0% ยิ่งสมมาตร")
            st.metric("Symmetry Index (สูงสุด)", f"{si_max:.2f}%", help="ยิ่งใกล้ 0% ยิ่งสมมาตร")

        # ปุ่มดาวน์โหลดไฟล์ CSV
        st.divider()
        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูลมุมเข่ารายเฟรม (.CSV)",
            data=csv_data,
            file_name="gait_knee_angles.csv",
            mime="text/csv"
        )
