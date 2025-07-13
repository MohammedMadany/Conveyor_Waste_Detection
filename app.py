import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
from collections import Counter
import os
import glob
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import shutil

# Load the trained YOLOv8 model for plastic waste detection
model = YOLO(r'E:/My Courses/CV Projects/01- Optical detection of plastic waste/model/Conveyor_Waste_Detection.pt')

# Configure Streamlit page with a professional layout and recycling icon
st.set_page_config(
    page_title="Conveyor Waste Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional and attractive design with Font Awesome icons
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa 0%, #f9f9f9 100%);
        padding: 20px 40px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #27ae60;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .header h1 {
        margin: 0;
        font-size: 2.2em;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header .tagline {
        font-size: 0.9em;
        margin-left: 10px;
        opacity: 0.8;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .stButton>button:hover {
        background-color: #219653;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-color: #34495e;
        color: white;
        padding: 15px;
    }
    .sidebar .stRadio label {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 0;
        color: white;
    }
    .sidebar .stRadio input[type="radio"] {
        margin-right: 8px;
    }
    .stProgress > div > div > div > div {
        background-color: #27ae60;
    }
    .footer {
        text-align: center;
        padding: 10px;
        background-color: #34495e;
        color: white;
        position: relative;
        width: 100%;
        font-size: 12px;
        margin-top: 20px;
    }
    .error-message, .warning-message {
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: 500;
    }
    .error-message {
        background-color: #e74c3c;
        color: white;
    }
    .warning-message {
        background-color: #f39c12;
        color: white;
    }
    .spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 24px;
        color: #27ae60;
    }
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        color: white;
        font-size: 20px;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    /* Ensure text visibility on Home page */
    .stApp > div[data-testid="stVerticalBlock"] p {
        color: #333333; /* Dark gray for readability */
    }
    .stApp > div[data-testid="stVerticalBlock"] a {
        color: #27ae60; /* Green for links */
    }
    @media (max-width: 600px) {
        .main { padding: 10px; }
        .header h1 { font-size: 1.8em; }
        .stButton>button { padding: 8px 15px; font-size: 14px; }
        .footer { position: relative; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with branded title and tagline
st.markdown(
    """
    <div class='header'>
        <h1>Conveyor Waste Detection</h1>
        <span class='tagline'>Innovating Recycling with AI</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar with styled logo and navigation
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <img src='E:/My Courses/CV Projects/01- Optical detection of plastic waste/src/logo.png' alt='Conveyor Waste Detection Logo' 
             style='width: 100%; border-radius: 5px; max-width: 150px;' 
             onError="this.src='https://via.placeholder.com/150'; this.alt='Default Logo';">
        <h3 style='color: white; margin: 10px 0 0; font-size: 1.2em;'>Navigation</h3>
    </div>
    """,
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    " ",
    ["Home", "Image Detection", "Video Detection", "About"],
    format_func=lambda x: f"{x}"
)

# Helper function to process video frames
def process_video_frame(frame, confidence):
    """Process a single video frame using the YOLO model and return the plotted result."""
    try:
        results = model.predict(source=frame, conf=confidence, save=False)
        return results[0].plot()
    except Exception as e:
        st.markdown(f"<div class='error-message'>Error processing frame: {e}</div>", unsafe_allow_html=True)
        return frame

# Home Page
if page == "Home":
    st.markdown("<h2 style='text-align: center; color: #27ae60;'>Welcome to Our Solution</h2>", unsafe_allow_html=True)
    st.write(
        """
        This application harnesses the power of YOLOv8 to detect PET and PP plastics on conveyor belts, 
        driving sustainable recycling initiatives.  
        - **Dataset**: Sourced from [Kaggle](https://www.kaggle.com/datasets/islomjon/conveyor-waste-detection-dataset).  
        - **Model**: YOLOv8 Nano, trained for 80 epochs, with an mAP50 of 0.993.  
        - **Features**: Real-time detection for images and videos.  
        """
    )
    st.write("Explore the project on [GitHub](https://github.com/MohammedMadany/Conveyor_Waste_Detection).")
    pr_curve_path = r"E:\My Courses\CV Projects\01- Optical detection of plastic waste\src\runs\detect\conveyor_waste2\BoxPR_curve.png"
    if os.path.exists(pr_curve_path):
        st.image(pr_curve_path, caption="Precision-Recall Curve", use_container_width=True)
    else:
        st.markdown("<div class='warning-message'>Precision-Recall Curve image not found. Please ensure training output exists in the runs directory.</div>", unsafe_allow_html=True)

# Image Detection Page
elif page == "Image Detection":
    st.markdown("<h2 style='color: #27ae60;'>Image-Based Waste Detection</h2>", unsafe_allow_html=True)
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, help="Adjust the confidence threshold for detection.")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            if st.button("Detect Waste"):
                with st.spinner():
                    st.markdown("<div class='loading-overlay'><i class='fas fa-spinner fa-spin'></i> Analyzing...</div>", unsafe_allow_html=True)
                    try:
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        results = model.predict(image, conf=confidence, save=True)
                        save_dir = results[0].save_dir
                        st.write(f"Debug: Save directory = {save_dir}")
                        output_images = glob.glob(os.path.join(save_dir, "*.jpg"))
                        if output_images:
                            predicted_image = max(output_images, key=os.path.getmtime)
                            if os.path.exists(predicted_image):
                                st.image(predicted_image, caption="Detected Waste", use_container_width=True)
                            else:
                                st.markdown("<div class='warning-message'>Detected image file not found at: {predicted_image}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='warning-message'>No detection output found. Check the save directory.</div>", unsafe_allow_html=True)

                        class_counts = Counter(results[0].boxes.cls.cpu().numpy().astype(int))
                        st.markdown("### Detection Results")
                        for class_id, count in class_counts.items():
                            st.write(f"- {['PET', 'PP'][class_id]}: {count} objects")

                        if output_images and os.path.exists(predicted_image):
                            with open(predicted_image, "rb") as file:
                                st.download_button(
                                    label="Download Result",
                                    data=file,
                                    file_name=f"detected_{os.path.basename(uploaded_image.name)}",
                                    mime="image/jpeg"
                                )
                    except Exception as e:
                        st.markdown(f"<div class='error-message'>Error during detection: {e}</div>", unsafe_allow_html=True)
                    finally:
                        st.markdown("<style>.loading-overlay { display: none; }</style>", unsafe_allow_html=True)

# Video Detection Page
elif page == "Video Detection":
    st.markdown("<h2 style='color: #27ae60;'>Video-Based Waste Detection</h2>", unsafe_allow_html=True)
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, help="Adjust the confidence threshold for detection.")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        st.write(f"Debug: Temp directory created at {temp_dir}")
        temp_video_path = os.path.join(temp_dir, uploaded_video.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getvalue())
        st.write(f"Debug: Uploaded video path = {temp_video_path}")
        # Convert to bytes for Streamlit to serve
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes, format="video/mp4")  # Display uploaded video using bytes

        if st.button("Detect Waste"):
            with st.spinner():
                st.markdown("<div class='loading-overlay'><i class='fas fa-spinner fa-spin'></i> Processing...</div>", unsafe_allow_html=True)
                try:
                    cap = cv2.VideoCapture(temp_video_path)
                    if not cap.isOpened():
                        st.markdown("<div class='error-message'>Error opening video file.</div>", unsafe_allow_html=True)
                    else:
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        output_dir = "runs/detect/exp"
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                        os.makedirs(output_dir)
                        st.write(f"Debug: Output directory created at {output_dir}")
                        output_path = os.path.join(output_dir, f"detected_{uploaded_video.name}")

                        # Try a different codec for better compatibility
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Changed from mp4v to XVID
                        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

                        progress_bar = st.progress(0)
                        frame_count = 0
                        total_detections = Counter()

                        with ThreadPoolExecutor() as executor:
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                processed_frame = executor.submit(process_video_frame, frame, confidence).result()
                                out.write(processed_frame)
                                results = model.predict(frame, conf=confidence, save=False)
                                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                                total_detections.update(class_ids)
                                frame_count += 1
                                progress_bar.progress(frame_count / total_frames)

                        cap.release()
                        out.release()

                        st.write(f"Debug: Output video path = {output_path}")
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as f:
                                output_video_bytes = f.read()
                            st.video(output_video_bytes, format="video/mp4")  # Display detected video using bytes
                        else:
                            st.markdown("<div class='warning-message'>Output video file not found at: {output_path}</div>", unsafe_allow_html=True)

                        st.markdown("### Detection Results")
                        for class_id, count in total_detections.items():
                            st.write(f"- {['PET', 'PP'][class_id]}: {count} objects")

                        if os.path.exists(output_path):
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="Download Result",
                                    data=file,
                                    file_name=f"detected_{os.path.basename(uploaded_video.name)}",
                                    mime="video/mp4"
                                )

                    # Clean up directories if they exist
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)

                except Exception as e:
                    st.markdown(f"<div class='error-message'>Error during processing: {e}</div>", unsafe_allow_html=True)
                    if "cap" in locals():
                        cap.release()
                    if "out" in locals():
                        out.release()
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
                finally:
                    st.markdown("<style>.loading-overlay { display: none; }</style>", unsafe_allow_html=True)

# About Page
elif page == "About":
    st.markdown("<h2 style='color: #27ae60;'>About This Project</h2>", unsafe_allow_html=True)
    st.write(
        """
        This project leverages YOLOv8 to detect PET and PP plastics on conveyor belts, promoting automated recycling.  
        - **Author**: Mohammed Madany (@Madanyy_)  
        - **GitHub**: [Conveyor_Waste_Detection](https://github.com/MohammedMadany/Conveyor_Waste_Detection)  
        - **Dataset**: [Kaggle](https://www.kaggle.com/datasets/islomjon/conveyor-waste-detection-dataset)  
        - **Performance**: mAP50: 0.993, mAP50-95: 0.815  
        - **Date**: Developed on July 13, 2025, at 10:18 PM EEST  
        """
    )
    st.write("Showcasing advanced computer vision skills. Explore the [GitHub repository](https://github.com/MohammedMadany/Conveyor_Waste_Detection).")
    labels_path = r"E:\My Courses\CV Projects\01- Optical detection of plastic waste\src\runs\detect\conveyor_waste2\labels.jpg"
    if os.path.exists(labels_path):
        st.image(labels_path, caption="Label Distribution", use_container_width=True)
    else:
        st.markdown("<div class='warning-message'>Label Distribution image not found. Please ensure training output exists.</div>", unsafe_allow_html=True)

    with st.expander("Project Details"):
        st.markdown("<h3 style='color: #27ae60;'>Technical Overview</h3>", unsafe_allow_html=True)
        st.write(
            """
            - **Framework**: Streamlit for UI, YOLOv8 for detection.
            - **Training**: 80 epochs on a custom dataset.
            - **Goal**: Enhance recycling efficiency with AI.
            """
        )

    with st.expander("Contact Information"):
        st.markdown("<h3 style='color: #27ae60;'>Get in Touch</h3>", unsafe_allow_html=True)
        st.write(
            "Connect with me: "
            "[LinkedIn](https://www.linkedin.com/in/mohammed-madany-20b408224) | "
            "[Email](mailto:madanym066@gmail.com) | "
            "[GitHub](https://github.com/MohammedMadany)"
        )

# Footer with social media icons
st.markdown(
    """
    <div class='footer'>
        © 2025 Mohammed Madany. All rights reserved. 
        <a href='https://www.linkedin.com/in/mohammed-madany-20b408224' target='_blank'><i class='fab fa-linkedin'></i></a>
        <a href='mailto:madanym066@gmail.com' target='_blank'><i class='fas fa-envelope'></i></a>
        <a href='https://github.com/MohammedMadany' target='_blank'><i class='fab fa-github'></i></a>
    </div>
    """,
    unsafe_allow_html=True
)