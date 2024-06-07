from datetime import datetime
import face_recognition
import numpy as np
import cv2 as cv
import os
import time

# Directory path for training images
path = 'Photos'
list_of_files = os.listdir(path)
images = []
names = []

# Encoding images in array
def encodings(images):
    enclist = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        enclist.append(encode)
    return enclist

# Array of images and names
for cls in list_of_files:
    img = cv.imread(f'{path}/{cls}')
    images.append(img)
    names.append(os.path.splitext(cls)[0])  # [0] to remove extensions from names

KnownList = encodings(images)

# Mark presence in CSV file
def presence(name):
    with open('Presence.csv', 'r+') as f:
        data = f.readlines()
        nameList = set(line.split(',')[0] for line in data)
        if name not in nameList:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.write(f'{name},{dt}\n')

# Get input from camera
cap = cv.VideoCapture(0)

# Reduce the frame processing frequency, by analysinng nth frame
frame_interval = 10

# Store face locations and names
detected_faces = []

# Check FPS every 'fps_check_interval' frames
fps_check_interval = 25

# Target FPS (can be adjusted based on requirements)
target_fps = 20

# Moving average parameters
fps_window = 50
fps_values = []

# Parameters for consecutive detection
consecutive_frames_threshold = 3
consecutive_frames = 0
prev_name = None

frame_count = 0
overall_t1 = time.time()
t1 = time.time()

while True:
    frame_count += 1
    success, img = cap.read()

    if not success:
        break

    # Process every nth frame
    if frame_count % frame_interval == 0:
        # Reduce size to 1/4th to speed up
        small_img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_img = cv.cvtColor(small_img, cv.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_img)
        face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)
        detected_faces.clear()
        
        # Compare faces
        for encode_face, face_loc in zip(face_encodings, face_locations):
            # tolerance value - lower the stricter
            matches = face_recognition.compare_faces(KnownList, encode_face, tolerance=0.4)
            
            # face_distances is array of distances of input from known images. Min needed.
            face_distances = face_recognition.face_distance(KnownList, encode_face)

            match_index = np.argmin(face_distances)
            if matches[match_index]:
                name = names[match_index]
                top, right, bottom, left = face_loc
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                
                detected_faces.append((name, left, top, right, bottom))
                presence(name)

    # Draw rectangles and names on every frame
    for name, left, top, right, bottom in detected_faces:
        cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv.rectangle(img, (left, top), (right, top - 35), (0, 255, 0), cv.FILLED)
        cv.putText(img, name, (left + 10, top - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow('Detections', img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

    # Calculate and update FPS using moving average
    if frame_count % fps_check_interval == 0:
        t2 = time.time()
        current_fps = fps_check_interval / (t2 - t1)
        fps_values.append(current_fps)
        t1 = t2         # Reset time for the next interval

        if len(fps_values) > fps_window:
            fps_values.pop(0)

        average_fps = sum(fps_values) / len(fps_values)
        print(f"Current FPS: {average_fps:.2f}")
        print(f"Current frame interval: {frame_interval:.2f}")

        # Adjust frame_interval based on target FPS with finer control
        if average_fps < target_fps - 1:
            frame_interval = min(frame_interval + 1, 60)  # Increase interval, but cap at 30
        elif average_fps > target_fps + 1:
            frame_interval = max(frame_interval - 1, 1)  # Decrease interval, but minimum is 1

cap.release()
cv.destroyAllWindows()

overall_t2 = time.time()
elapsed_time = overall_t2 - overall_t1

final_fps = frame_count / elapsed_time
print(f"\nFinal Average FPS: {final_fps:.2f}")