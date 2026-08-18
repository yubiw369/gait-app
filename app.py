import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import plotly.graph_objects as go

# 1. PAGE CONFIG & CUSTOM CSS
st.set_page_config(page_title="Gait Analysis System", page_icon="🦵", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #F8F9FA;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E9ECEF;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0F52BA !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🦵 ระบบวิเคราะห์ท่าเดินจากวิดีโอ (Video Gait Analysis System)")
st.caption("พัฒนาขึ้นเพื่อการวิเคราะห์มุมข้อต่อชีวกลศาสตร์ และดัชนีความสมมาตร (Symmetry Index)")
st.divider()

# 2. CORE MATH FUNCTIONS
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

def calculate_symmetry_index(x_left, x_right):
    if (x_left + x_right) == 0:
        return 0.0
    return (2 * abs(x_right - x_left) / (x_right + x_left)) * 100.0

# 3. SIDEBAR
st.sidebar.header("⚙️ การตั้งค่าระบบ")
input_type = st.sidebar.radio("เลือกแหล่งข้อมูล:", ["📹 อัปโหลดไฟล์วิดีโอ (Video Upload)", "🎲 จำลองข้อมูลทดสอบ (Demo Data)"])
window_size = st.sidebar.slider("ขนาด Window (Moving Average):", 1, 15, 5)

# 4. DATA PROCESSING
df_result = None

if input_type == "📹 อัปโหลดไฟล์วิดีโอ (Video Upload)":
    uploaded_video = st.file_uploader("อัปโหลดคลิปวิดีโอการเดิน (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        frames_data = []
        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                l_hip, l_knee, l_ankle = [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
                r_hip, r_knee, r_ankle = [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
                
                frames_data.append({
                    'Frame': frame_count,
                    'Left_Knee': calculate_angle(l_hip, l_knee, l_ankle),
                    'Right_Knee': calculate_angle(r_hip, r_knee, r_ankle)
                })
            
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"กำลังประมวลผลวิดีโอ เฟรมที่: {frame_count}/{total_frames}")
            
        cap.release()
        pose.close()
        status_text.success("✅ ประมวลผลวิดีโอเสร็จสิ้น!")
        
        if len(frames_data) > 0:
            df_result = pd.DataFrame(frames_data)
        else:
            st.error("❌ ไม่พบโครงกระดูกมนุษย์ในวิดีโอ กรุณาตรวจสอบมุมกล้องและแสงสว่าง")
            st.stop()
    else:
        st.info("👋 กรุณาอัปโหลดไฟล์วิดีโอ หรือเลือก 'จำลองข้อมูลทดสอบ' เพื่อดูตัวอย่าง")
        st.stop()
else:
    num_frames = 100
    time_steps = np.linspace(0, 4 * np.pi, num_frames)
    np.random.seed(42)
    df_result = pd.DataFrame({
        'Frame': np.arange(1, num_frames + 1),
        'Left_Knee': 35 + 25 * np.sin(time_steps) + np.random.normal(0, 2, num_frames),
        'Right_Knee': 33 + 23 * np.sin(time_steps + 0.15) + np.random.normal(0, 2, num_frames)
    })

# 5. SMOOTHING & METRICS
df_result['Left_Knee_Smooth'] = df_result['Left_Knee'].rolling(window=window_size, min_periods=1).mean()
df_result['Right_Knee_Smooth'] = df_result['Right_Knee'].rolling(window=window_size, min_periods=1).mean()

max_left = df_result['Left_Knee_Smooth'].max()
max_right = df_result['Right_Knee_Smooth'].max()
si_value = calculate_symmetry_index(max_left, max_right)

# 6. DASHBOARD
tab1, tab2, tab3 = st.tabs(["📊 สรุปผลการวิเคราะห์", "📖 สูตรคณิตศาสตร์ที่ใช้", "📥 ส่งออกรายงาน"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("เข่าซ้ายงอสูงสุด", f"{max_left:.1f}°")
    col2.metric("เข่าขวางอสูงสุด", f"{max_right:.1f}°")
    col3.metric("Symmetry Index", f"{si_value:.2f}%")
    
    if si_value < 10.0:
        col4.success("ปกติ (Symmetric)")
    else:
        col4.warning("ไม่สมมาตร (Asymmetric)")
        
    st.divider()
    
    # Plotly Interactive Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_result['Frame'], y=df_result['Left_Knee_Smooth'], mode='lines', name='เข่าซ้าย (Left Knee)', line=dict(color='#1E88E5', width=3)))
    fig.add_trace(go.Scatter(x=df_result['Frame'], y=df_result['Right_Knee_Smooth'], mode='lines', name='เข่าขวา (Right Knee)', line=dict(color='#E53935', width=3, dash='dash')))
    fig.update_layout(title="📈 กราฟเปรียบเทียบการเคลื่อนไหวข้อเข่า (Gait Cycle)", xaxis_title="เฟรมวิดีโอ (Frame)", yaxis_title="มุมงอข้อเข่า (องศา)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📐 ทฤษฎีคณิตศาสตร์เบื้องหลังการคำนวณ")
    st.write("1. **ตรีโกณมิติ ($\operatorname{atan2}$)**")
    st.latex(r"\theta = \left| \operatorname{atan2}(y_c - y_b, x_c - x_b) - \operatorname{atan2}(y_a - y_b, x_a - x_b) \right| \times \frac{180}{\pi}")
    st.write("2. **Moving Average (SMA)**")
    st.latex(r"SMA_t = \frac{1}{N} \sum_{i=0}^{N-1} x_{t-i}")
    st.write("3. **Symmetry Index (SI)**")
    st.latex(r"SI = \frac{2 \cdot |X_{\text{Right}} - X_{\text{Left}}|}{X_{\text{Right}} + X_{\text{Left}}} \times 100\%")

with tab3:
    st.dataframe(df_result, use_container_width=True)
    st.download_button("📄 ดาวน์โหลดไฟล์ CSV", data=df_result.to_csv(index=False).encode('utf-8'), file_name="gait_analysis_results.csv", mime="text/csv")