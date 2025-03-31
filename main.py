from datetime import datetime
import os, logging
import face_recognition
import numpy as np
import cv2 as cv

# Directory path for training images
path = 'Photos'
list_of_files = os.listdir(path)
images = []
names = []

# Create logs directory if it doesn't exist
LOGS_DIR = 'logs'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
log_filename = f"prosence_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, log_filename)),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('FaceRecognition')
logger.info(f"Logging to file: {log_filename}")

# Create a new CSV file in OUTPUT_DIR on every execution
csv_filename = f"Presence_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
csv_path = os.path.join(OUTPUT_DIR, csv_filename)

with open(csv_path, 'w') as f:
    f.write("Name,Time\n")

logger.info(f"Presence CSV initialized: {csv_filename}")

# Encoding images in array
def encodings(images):
    logger.info("Starting image encodings.")
    enclist = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        enclist.append(encode)
    logger.info("Image encodings completed.")
    return enclist

# Array of images and names
logger.info("Loading training images and names.")
for cls in list_of_files:
    img = cv.imread(f'{path}/{cls}')
    images.append(img)
    names.append(os.path.splitext(cls)[0])  # [0] to remove extensions from names
logger.info(f"Loaded {len(images)} images.")

KnownList = encodings(images)

# Mark presence in CSV file
marked_names = set()
def presence(name):
    global marked_names
    if name in marked_names:
        return
    
    logger.info(f"Marking presence for: {name}")
    with open(csv_path, 'r+') as f:
        data = f.readlines()
        nameList = {line.split(',')[0] for line in data}

        if name not in nameList:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.write(f'{name},{dt}\n')
            logger.info(f"{name} presence recorded at {dt}")
            marked_names.add(name)

def recognize_faces(frame, known_face_encodings, known_face_names, tolerance=0.5):
    """Recognizes faces in a frame and returns detected face names."""
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    detected_faces = []

    for encode_face, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, encode_face, tolerance=tolerance)
        name = "unknown"

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_face_encodings, encode_face))
            name = known_face_names[match_index]
            presence(name)

        top, right, bottom, left = face_loc
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        detected_faces.append((name, left, top, right, bottom))
    return detected_faces, frame

def load_known_faces(photos_dir):
    images = []
    names = []
    for filename in os.listdir(photos_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv.imread(os.path.join(photos_dir, filename))
            if image is not None:
                images.append(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                names.append(os.path.splitext(filename)[0])
    encodings = [face_recognition.face_encodings(img)[0] for img in images if face_recognition.face_encodings(img)]
    return encodings, names