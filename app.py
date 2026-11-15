import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
from google import genai
import os
import time
from dotenv import dotenv_values
from PIL import Image
import tempfile
from pathlib import Path
from html import escape
from gemini_diagnostics import explain_gemini_error
from frame_display import frame_data_uri
from video_frames import sample_video_frames

# ================= 1. SETUP & CONFIG =================
process_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip() or None
dotenv_api_key = (
    (dotenv_values(Path(__file__).with_name(".env")).get("GEMINI_API_KEY") or "").strip()
    or None
)
api_key = process_api_key or dotenv_api_key
api_key_source = "process environment" if process_api_key else ".env file"
GEMINI_MODEL = "gemini-3.5-flash-lite"

st.set_page_config(layout="wide", page_title="Sentinel AI - Command Center", page_icon="🧿")

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
    .frame-image { margin: 0 0 1rem; text-align: center; }
    .frame-image img { display: block; width: 100%; height: auto; object-fit: contain; border-radius: 8px; }
    .frame-image figcaption { color: #94a3b8; padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ================= 2. MODEL LOADING =================
@st.cache_resource
def load_models():
    ym = YOLO('best.pt') 
    pm = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return ym, pm

@st.cache_resource
def load_gemini_client(key):
    return genai.Client(api_key=key)

if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to the .env file in the project directory.")
    st.stop()

ym, pm = load_models()
gm = load_gemini_client(api_key)
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
    st.title("🛡️ Sentinel AI")
    st.markdown("### System Specs")
    st.info(f"🧠 Gemini: `{GEMINI_MODEL}`\n\n👀 Vision: YOLOv8 (Custom)\n\n🦴 Spatial: MediaPipe 3D")
    st.caption("Custom YOLO detects people. Gemini's visual assessment analyzes the scene.")
    st.caption(f"API key source: {api_key_source} (value hidden)")
    if process_api_key and dotenv_api_key and process_api_key != dotenv_api_key:
        st.warning(
            "The process environment key overrides the .env key. Remove the stale "
            "GEMINI_API_KEY from the terminal/VS Code environment and restart if "
            "you intend to use the .env key."
        )
    if st.button("Check Gemini model access", key="check_gemini_access"):
        try:
            available_model = None
            for model in gm.models.list():
                listed_name = (model.name or "").rsplit("/", 1)[-1]
                if listed_name == GEMINI_MODEL:
                    available_model = model
                    break

            if available_model is None:
                st.error(
                    f"`{GEMINI_MODEL}` is not listed for this key's Google project. "
                    "Check that the key belongs to the intended project and that "
                    "the model is available to it."
                )
            elif "generateContent" in (available_model.supported_actions or []):
                st.success(f"`{GEMINI_MODEL}` is listed and supports generateContent.")
            else:
                st.error(
                    f"`{GEMINI_MODEL}` is listed, but generateContent is not shown "
                    "as a supported action for this key."
                )
        except Exception as error:
            st.error(explain_gemini_error(error, GEMINI_MODEL))
    st.markdown("---")
    st.markdown("### Development Team")
    st.markdown("👨‍💻 **Anant Gupta**\n\n👨‍💻 **Saksham Pandey**\n\n👨‍💻 **Ravi Kumar Saini**\n\n👨‍💻 **Krishna Karma**")
    st.markdown("---")
    st.success("Local models loaded 🟢")

# ================= 4. MAIN DASHBOARD =================
st.markdown("<h1>🧿 Sentinel AI Command Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Video scene analysis with YOLOv8, MediaPipe, and Gemini</p>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric(label="Runtime Status", value="Ready", delta="Local models loaded")
m2.metric(label="Analysis Modules", value="3/3", delta="Initialized")
yolo_class_names = [str(name).title() for name in ym.names.values()]
m3.metric(label="YOLO model classes", value=", ".join(yolo_class_names) or "None")
st.caption("YOLO model classes are the labels configured in best.pt, not live detections.")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔴 Live Surveillance Stream", "📁 Archive Video Analysis"])

STRICT_PROMPT = """Analyze the supplied sampled frames from a security camera.
Use only evidence visible in the supplied frames. Do not infer identity, intent, or
events that are not shown. These are sparse samples: state uncertainty when an
event may occur between frames. Do not call a scene safe just because no threat is
visible. Return three short sections:

EVENT SUMMARY: Describe visible actions and objects in 2 sentences.
THREAT ASSESSMENT: Describe only clearly visible concerns, or state that no clear
threat is visible in the sampled frames.
CONCLUSION: State whether the samples show a potential concern or are inconclusive;
do not certify the scene as safe."""

def capture_live_samples(capture, preview, progress, duration=3.0, sample_count=5):
    frames = []
    start_time = time.time()
    next_preview_time = start_time
    sample_times = [duration * (index + 0.5) / sample_count for index in range(sample_count)]

    while time.time() - start_time < duration and len(frames) < sample_count:
        success, frame = capture.read()
        if not success:
            time.sleep(0.01)
            continue
        elapsed = time.time() - start_time
        if elapsed >= sample_times[len(frames)]:
            frames.append(frame.copy())
            progress.progress(int((len(frames) / sample_count) * 100))
        if elapsed >= next_preview_time:
            preview.markdown(frame_html(frame, "Live camera preview"), unsafe_allow_html=True)
            next_preview_time = elapsed + 0.2
    return frames


def frame_html(frame_bgr, caption):
    safe_caption = escape(caption)
    return (
        '<figure class="frame-image">'
        f'<img src="{frame_data_uri(frame_bgr)}" alt="{safe_caption}">'
        f"<figcaption>{safe_caption}</figcaption>"
        "</figure>"
    )


def render_frame_gallery(annotated_frames):
    st.markdown("### 🎞️ Analyzed Frames")
    columns = st.columns(min(3, len(annotated_frames)))
    for index, frame in enumerate(annotated_frames):
        with columns[index % len(columns)]:
            st.markdown(
                frame_html(frame, f"Sample {index + 1} of {len(annotated_frames)}"),
                unsafe_allow_html=True,
            )

def generate_report(frames):
    if not frames:
        raise ValueError("No video frames were available for analysis.")

    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    response = gm.models.generate_content(
        model=GEMINI_MODEL,
        contents=[STRICT_PROMPT] + pil_images,
    )
    if not response.text or not response.text.strip():
        raise RuntimeError("Gemini returned an empty analysis.")
    return response.text

# --- TAB 1: LIVE 3-SEC CAPTURE ---
with tab1:
    st.markdown("### 🎥 Camera Feed")
    cam_placeholder = st.empty()
    cam_placeholder.info("Camera preview appears here after capture starts.")
    
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        start_btn = st.button("🚨 Initialize 3-Sec Capture & Analyze", key="live_btn")
    
    if start_btn:
        cap = cv2.VideoCapture(0)
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        if not cap.isOpened():
            st.error("Camera could not be opened. Check permissions.")
        else:
            try:
                status_text.info("📡 Capturing a 3-second sequence...")
                raw_frames = capture_live_samples(cap, cam_placeholder, progress_bar)
                cap.release()

                if raw_frames:
                    status_text.info("⚙️ Running local vision models...")
                    annotated_frames = [process_frame(frame) for frame in raw_frames]
                    render_frame_gallery(annotated_frames)
                    st.markdown("### 🧠 Incident Report")
                    status_text.info("Sending sampled frames to Gemini...")
                    try:
                        report = generate_report(raw_frames)
                        status_text.success("✅ Analysis complete.")
                        st.markdown(report)
                    except Exception as error:
                        status_text.error(explain_gemini_error(error, GEMINI_MODEL))
                else:
                    st.error("No camera frames were captured.")
            except Exception as error:
                status_text.error(f"Live analysis failed: {error}")
                if cap.isOpened():
                    cap.release()

# --- TAB 2: UPLOAD VIDEO ---
with tab2:
    st.markdown("### 📁 Upload External Footage")
    uploaded_video = st.file_uploader("Select Video File (MP4, AVI)", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        if st.button("Start Diagnostics", key="upload_btn"):
            video_path = None
            cap = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as video_file:
                    video_file.write(uploaded_video.getvalue())
                    video_path = video_file.name

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError("OpenCV could not open this video. Try MP4 (H.264) or AVI.")
                status_v_text = st.empty()
                status_v_text.info("⚙️ Sampling frames across the full video...")
                
                raw_frames = sample_video_frames(cap)
                cap.release()
                
                if not raw_frames:
                    st.error("No frames could be read from the uploaded video.")
                else:
                    status_v_text.info("⚙️ Running local vision models...")
                    annotated_frames = [process_frame(frame) for frame in raw_frames]
                    render_frame_gallery(annotated_frames)
                    st.markdown("### 🧠 Incident Report")
                    status_v_text.info("Sending sampled frames to Gemini...")
                    try:
                        report = generate_report(raw_frames)
                        status_v_text.success("✅ Full-video analysis complete.")
                        st.markdown(report)
                    except Exception as error:
                        status_v_text.error(explain_gemini_error(error, GEMINI_MODEL))
            except Exception as error:
                st.error(f"Video analysis failed: {error}")
            finally:
                if cap is not None and cap.isOpened():
                    cap.release()
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)