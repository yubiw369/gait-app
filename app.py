import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import tempfile
import os

# ==========================================
# แก้ไขปัญหา AttributeError: mp.solutions.pose
# ==========================================
import mediapipe as mp
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except (ImportError, ModuleNotFoundError):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

# ==========================================
# ตั้งค่าหน้าเว็บ Streamlit
# ==========================================
st.set_page_config(
    page_title="ระบบวิเคราะห์ท่าเดินจากวิดีโอ",
    page_icon="🦶",
    layout="wide"
)

st.title("🦶 ระบบวิเคราะห์ท่าเดินจากวิดีโอ (Video Gait Analysis System)")
st.caption("พัฒนาขึ้นเพื่อการวิเคราะห์มุมข้อต่อชีวกลศาสตร์ และดัชนีความสมมาตร (Symmetry Index)")

# ==========================================
# ฟังก์ชันคำนวณมุมระหว่าง 3 จุด
# ==========================================
def calculate_angle(a, b, c):
    a = np.array(a) # จุดเริ่ม (เช่น สะโพก)
    b = np.array(b) # จุดหมุน (เช่น เข่า)
    c = np.array(c) # จุดปลาย (เช่น ข้อเท้า)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# ==========================================
# ส่วนการทำงานหลัก (Upload & Processing)
# ==========================================
uploaded_file = st.file_uploader("อัปโหลดคลิปวิดีโอการเดิน (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 1. สร้างไฟล์ชั่วคราวเพื่อให้อ่านวิดีโอผ่าน OpenCV ได้
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.info("อัปโหลดวิดีโอเรียบร้อยแล้ว กำลังเริ่มประมวลผล...")

    # 2. เปิดไฟล์วิดีโอด้วย OpenCV
    cap = cv2.VideoCapture(video_path)
    
    frames_data = []
    frame_count = 0

    st_frame = st.empty()

    # 3. เริ่มประมวลผลด้วย MediaPipe Pose
    with st.spinner('กำลังวิเคราะห์ท่าเดินด้วย MediaPipe...'):
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                # แปลง BGR เป็น RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # ประมวลผลหาจุดโครงสร้างร่างกาย
                results = pose.process(image)
                
                image.flags.writeable = True
                
                # คำนวณมุมหากพบโครงสร้างร่างกาย
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # ดึงพิกัดขาซ้าย
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    # ดึงพิกัดขาขวา
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    # คำนวณมุมเข่า
                    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    
                    # บันทึกข้อมูลเฟรม
                    frames_data.append({
                        'Frame': frame_count,
                        'Left Knee Angle': left_knee_angle,
                        'Right Knee Angle': right_knee_angle
                    })

                    # วาดเส้น Skeleton บนภาพ
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                # แสดงวิดีโอแบบเรียลไทม์ (ข้ามเฟรมเพื่อความเร็วบน Cloud)
                if frame_count % 2 == 0:
                    st_frame.image(image, channels="RGB", use_container_width=True)

    # ปิดและลบไฟล์ชั่วคราว
    cap.release()
    try:
        os.remove(video_path)
    except Exception:
        pass

    st.success("วิเคราะห์วิดีโอเสร็จสิ้น!")

    # ==========================================
    # 4. แสดงผลลัพธ์ข้อมูลและกราฟ
    # ==========================================
    if frames_data:
        df = pd.DataFrame(frames_data)
        
        st.subheader("📊 กราฟแสดงการเปลี่ยนแปลงมุมเข่า (Knee Joint Angle)")
        fig = px.line(
            df, 
            x='Frame', 
            y=['Left Knee Angle', 'Right Knee Angle'],
            labels={'value': 'มุม (องศา)', 'variable': 'ข้าง'},
            title="การเปรียบเทียบมุมเข่าซ้ายและขวาตลอดช่วงการเดิน"
        )
        st.plotly_chart(fig, use_container_width=True)

        # สรุปค่าสถิติ
        col1, col2 = st.columns(2)
        with col1:
            st.metric("มุมเข่าซ้ายเฉลี่ย", f"{df['Left Knee Angle'].mean():.2f}°")
            st.metric("มุมเข่าซ้ายสูงสุด", f"{df['Left Knee Angle'].max():.2f}°")
        with col2:
            st.metric("มุมเข่าขวาเฉลี่ย", f"{df['Right Knee Angle'].mean():.2f}°")
            st.metric("มุมเข่าขวาสูงสุด", f"{df['Right Knee Angle'].max():.2f}°")
