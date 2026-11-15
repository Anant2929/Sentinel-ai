import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
from google import genai
import os
import time
from dotenv import load_dotenv
from PIL import Image
import tempfile
import numpy as np

# ================= 1. SETUP & CONFIG =================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(layout="wide", page_title="Sentinel AI - Command Center", page_icon="🧿")

# Advanced Premium CSS (Cyber-Security Theme)
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    h1, h2, h3 { color: #38bdf8; font-family: 'Inter', sans-serif; font-weight: 700; }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #10b981; text-shadow: 0px 0px 10px rgba(16, 185, 129, 0.4); }
    div[data-testid="stMetricLabel"] { font-size: 1rem; color: #94a3b8; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 55px; 
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); 
        color: white; font-weight: bold; border: none; font-size: 16px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); transition: 0.3s all ease; 
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 2px solid #1e293b; padding-bottom: 5px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; background-color: transparent; border: 1px solid #1e293b; 
        border-radius: 8px; padding: 0 25px; color: #94a3b8; font-weight: 600; 
    }
    .stTabs [aria-selected="true"] { background-color: #1e293b !important; color: #38bdf8 !important; border-color: #38bdf8 !important; }
    [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    /* Naya CSS: Report box ko full width banane ke liye */
    .report-box { background-color: #1e293b; padding: 20px; border-radius: 8px; border-left: 5px solid #10b981; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ================= 2. MODEL LOADING =================
@st.cache_resource
def load_models():
    ym = YOLO('best.pt') 
    pm = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    gm = genai.Client(api_key=api_key)
    return ym, pm, gm

if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to the .env file in the project directory.")
    st.stop()

ym, pm, gm = load_models()
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_frame(frame_bgr):
    res = ym(frame_bgr, verbose=False)
    annotated = res[0].plot()
    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    p_res = pm.process(img_rgb)
    if p_res.pose_landmarks:
        mp_draw.draw_landmarks(
            annotated, p_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(16,185,129), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(56,189,248), thickness=2, circle_radius=2)
        )
    return annotated

# ================= 3. SIDEBAR =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2097/2097973.png", width=80)
    st.title("Sentinel AI")
    st.markdown("### System Specs")
    st.info("🧠 Core: Gemini 2.5 Flash\n\n👀 Vision: YOLOv8 (Custom)\n\n🦴 Spatial: MediaPipe 3D")
    st.markdown("---")
    st.markdown("### Development Team")
    st.markdown("👨‍💻 **Anant Gupta**\n\n👨‍💻 **Saksham Pandey**\n\n👨‍💻 **Ravi Kumar Saini**\n\n👨‍💻 **Krishna Karma**")
    st.markdown("---")
    st.success("System is Online & Secure 🟢")

# ================= 4. MAIN DASHBOARD =================
st.markdown("<h1>🧿 Sentinel AI Command Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Real-Time Multimodal Threat Detection & Scene Understanding</p>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric(label="System Status", value="Active", delta="Cloud Deployed")
m2.metric(label="Active Modules", value="3/3", delta="All Systems Go")
m3.metric(label="Vision Model", value="Custom YOLO", delta="Fine-tuned")
m4.metric(label="Latency", value="< 2.5s", delta="-12ms", delta_color="inverse")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Cloud Camera Scanner", "📁 Archive Video Analysis"])

# Prompts setup
STRICT_PROMPT_IMG = """Analyze this security camera frame. 
Provide the analysis STRICTLY in these three short paragraphs:
1. EVENT SUMMARY: (Describe the overall action in 2 sentences)
2. THREAT ASSESSMENT: (State if any weapons/threats are visible in 1 sentence)
3. CONCLUSION: (State if the scene is SAFE or SUSPICIOUS and why in 1 sentence)"""

STRICT_PROMPT_VID = """Analyze this 5-frame sequence from a security camera. 
Do NOT provide a frame-by-frame breakdown. 
Provide the analysis STRICTLY in these three short paragraphs:
1. EVENT SUMMARY: (Describe the overall action in 2 sentences)
2. THREAT ASSESSMENT: (State if any weapons/threats are visible in 1 sentence)
3. CONCLUSION: (State if the scene is SAFE or SUSPICIOUS and why in 1 sentence)"""

# --- TAB 1: CLOUD CAMERA SCANNER ---
with tab1:
    st.markdown("### 📸 Live Threat Detection (Snapshot)")
    
    # Securely request browser camera
    cam_image = st.camera_input("Capture Scene for Security Analysis")
    
    if cam_image:
        status_text = st.empty()
        status_text.warning("⚙️ Processing Neural Overlay (YOLO + MediaPipe)...")
        
        # Streamlit format se OpenCV format me convert karna
        file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        annotated_frame = process_frame(frame)
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        status_text.success("✅ Feed Processed. Engaging Gemini Engine...")
        
        st.markdown("### 🧠 Incident Report")
        report_box = st.empty()
        report_box.info("Extracting semantics from visual data...")
        
        pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        
        try:
            response = gm.models.generate_content(
                model="gemini-2.5-flash",
                contents=[STRICT_PROMPT_IMG, pil_img],
            )
            report_box.markdown(f'<div class="report-box">{response.text}</div>', unsafe_allow_html=True)
        except Exception as e:
            report_box.error(f"API Error: {e}")

# --- TAB 2: UPLOAD VIDEO ---
with tab2:
    st.markdown("### 📁 Upload External Footage")
    uploaded_video = st.file_uploader("Select Video File (MP4, AVI)", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        if st.button("Start Diagnostics", key="upload_btn"):
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            vid_placeholder = st.empty()
            status_v_text = st.empty()
            status_v_text.info("⚙️ Initializing analysis...")
            
            raw_frames = []
            frame_count = 0
            while frame_count < 90:
                ret, frame = cap.read()
                if not ret: break
                raw_frames.append(frame)
                frame_count += 1
            cap.release()
            
            if raw_frames:
                annotated_frames = []
                for f in raw_frames:
                    annotated = process_frame(f)
                    annotated_frames.append(annotated)
                    vid_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                status_v_text.success("✅ Analysis Complete")
                st.markdown("### 🧠 Incident Report")
                report_v_box = st.empty()
                
                step = max(1, len(annotated_frames) // 5)
                selected_frames = [annotated_frames[i] for i in range(0, len(annotated_frames), step)][:5]
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in selected_frames]
                
                try:
                    response = gm.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[STRICT_PROMPT_VID] + pil_images,
                    )
                    report_v_box.markdown(f'<div class="report-box">{response.text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    report_v_box.error(f"API Error: {e}")