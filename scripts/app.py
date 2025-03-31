import streamlit as st
import main
import cv2
import time
import pandas as pd
import os
from datetime import datetime

st.title("📸 Face Recognition Attendance System")

photos_dir = st.sidebar.text_input("Photos Directory", "Photos")
tolerance = st.sidebar.slider("Face Match Tolerance", 0.1, 0.9, 0.5)

# Load known faces with error handling
if os.path.exists(photos_dir) and os.listdir(photos_dir):
    try:
        known_face_encodings, known_face_names = main.load_known_faces(photos_dir)
    except Exception as e:
        st.error(f"Error loading faces: {e}")
        st.stop()
else:
    st.error(f"Error: photos directory {photos_dir} does not exist or is empty.")
    st.stop()

known_face_encodings, known_face_names = main.load_known_faces(photos_dir)

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
detected_faces_placeholder = st.empty()
csv_display_placeholder = st.empty()

# Frame processing frequency (analysing nth frame); n<=30 to prevent lags
frame_interval = 18
max_frm_int = 30

# Check FPS every n frames
fps_check_interval = 30

# Target FPS for overall video
target_fps = 18
text0 = f"Set Target: {target_fps} fps"
text1 = f"Current fps: {frame_interval:.2f}"
text2 = f"Analysing 1/{frame_interval}"

lag = (fps_check_interval / target_fps) * 1.1

# Moving average parameters - takes avg of prev n fps before reporting
fps_window = 30
fps_values = []

frame_count = 0
overall_t1 = time.time()
t1 = time.time()

# Initialize detected_faces to persist across frames
detected_faces = []
latest_csv = main.csv_path

while True:
    frame_count += 1
    ret, img = cap.read()

    if not ret:
        st.error("Failed to capture image")
        break

    cv2.rectangle(img, (5, 10), (200, 130), (0, 0, 0), 2)
    cv2.putText(img, text0, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, text1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, text2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"{lag:.2f}s for {fps_check_interval} frames", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Process every nth frame
    if frame_count % frame_interval == 0:
        detected_faces, img = main.recognize_faces(img, known_face_encodings, known_face_names, tolerance)

    # Draw persisted face rectangles and names (even on skipped frames)
    for name, left, top, right, bottom in detected_faces:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Calculate and update FPS using moving average
    if frame_count % fps_check_interval == 0:
        t2 = time.time()
        current_fps = fps_check_interval / (t2 - t1)
        fps_values.append(current_fps)
        lag = t2 - t1
        t1 = t2

        if len(fps_values) > fps_window:
            fps_values.pop(0)

        average_fps = sum(fps_values) / len(fps_values)
        text1 = f"Current fps: {average_fps:.2f}"
        text2 = f"Analysing 1/{frame_interval} frames"

        if (average_fps < target_fps - 1) or (lag > ((fps_check_interval / target_fps) * 1.1)):
            frame_interval = min(frame_interval + 1, max_frm_int)

        elif (average_fps > target_fps + 1):
            frame_interval = max(frame_interval - 1, 1)

    frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
    detected_faces_placeholder.text(f"Detected Person: {', '.join(face[0] for face in detected_faces)}")

    # Display the latest CSV with wider width
    if latest_csv and os.path.exists(latest_csv):
        try:
            df = pd.read_csv(latest_csv)
            csv_display_placeholder.dataframe(df, column_config={"Name": st.column_config.Column(width="large"), "Time": st.column_config.Column(width="large")})
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        csv_display_placeholder.text("No CSV file found.")

    time.sleep(0.1)

cap.release()
st.success("Camera feed stopped.")