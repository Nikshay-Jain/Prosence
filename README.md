# Prosence - Face Recognition Attendance System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Installation Guide](#installation-guide)
5. [Usage Instructions](#usage-instructions)
6. [File Structure](#file-structure)
7. [Output Format](#output-format)
8. [Limitations](#limitations)
9. [Future Enhancements](#future-enhancements)
10. [Contributing](#contributing)
11. [Closing note](#closing-note)

## Project Overview
Prosence is an automated attendance system that uses facial recognition to identify individuals and record their presence with timestamps. Designed for security and attendance tracking applications, it's suitable for:
- Educational institutions
- Corporate offices
- Security systems (CCTV integration)
- Military surveillance drones

## Key Features
✔ Real-time face detection and recognition  
✔ Automated attendance recording with timestamps  
✔ Simple database management via image uploads  
✔ CSV-based attendance logs with dynamic file creation  
✔ High accuracy recognition using deep learning  
✔ Logging system for debugging and tracking attendance activity  

## Technology Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Core Framework | Python 3.8 | Base programming language |
| Face Detection | OpenCV 4.11+ | Real-time face detection |
| Face Recognition | face-recognition (dlib) | Feature extraction and matching |
| Build Tools | CMake | Library dependencies |
| Logging | Python Logging Module | Tracks system events |

## Installation Guide

### Prerequisites
- Python 3.8
- CMake (for dlib compilation)

### Step-by-Step Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Nikshay-Jain/Prosence.git
   cd Prosence
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

#### If on Windows:

3. Install dependencies:
   ```bash
   python -m pip install "C:\your\path\to\dlib-19.22.99-cp38-cp38-win_amd64.whl"
   pip install -r requirements.txt
   ```

#### If on Linux/Mac:
3. Install dependencies:
   ```bash
   pip install dlib
   pip install -r requirements.txt
   ```

## Usage Instructions

### Adding New Faces
1. Place high-quality frontal face images in `/Photos` directory
2. Name files as `Firstname Lastname.jpg` (e.g., `John Doe.jpg`)
3. Recommended image specifications:
   - Minimum 300×300 pixels
   - Well-lit frontal view
   - No obstructions (glasses/masks optional)

### Running the System
```bash
streamlit run scripts/app.py
```

## File Structure
```
Prosence/
├── venv/                           # Virtual environment
├── Photos/                         # Folder for face database
├── logs/                           # Logs directory (system events and attendance tracking)
|    └──prosence_<timestamp>.log
├── output/                         # Directory for dynamically generated CSV attendance files
|    └──Presence_<timestamp>.csv    # Attendance records (new file for each execution)
├── scripts/                        # Directory containing code files
|    ├──main.py                     # Main application script
|    └──app.py                      # Streamlit application
├── requirements.txt                # Dependency list
├── .gitignore                      # Gitignore file
└── README.md                       # This file
```

## Output Format
The system generates a dynamically named CSV file in the `output/` directory with the following format:
```csv
Name,Time
John Doe, 09:30:45
Jane Smith, 09:31:22
```
Each execution creates a new CSV file with a timestamp in the filename (`Presence_YYYY-MM-DD_HH-MM-SS.csv`).

## Logging System
- The system maintains logs in the `logs/` directory.
- Logs track face recognition events, attendance updates, and errors.
- Example log entries:
  ```
  [INFO] 2025-03-31 10:15:22 - Presence CSV initialized: Presence_2025-03-31_10-15-22.csv
  [INFO] 2025-03-31 10:16:00 - Marking presence for: John Doe
  [INFO] 2025-03-31 10:16:05 - John Doe presence recorded at 10:16:05
  ```

## Limitations
⚠ Performance decreases in low-light conditions  
⚠ Side-profile faces may not be recognized  
⚠ Requires clear images for registration  
⚠ CSV file resets with each program execution  

## Future Enhancements
- [ ] Cloud synchronization for attendance records
- [ ] Mobile app integration
- [ ] Support for masked faces
- [ ] Live video streaming capability
- [ ] Database encryption

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## Closing Note
The Face Recognition System presented here utilizes OpenCV and face-recognition libraries to detect and recognize human faces accurately. By following the instructions outlined in this documentation, users can seamlessly add new faces to the system's database, track the presence of faces, and generate attendance records dynamically while maintaining logs for debugging and auditing purposes.