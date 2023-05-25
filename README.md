# Prosence
### Face Recognition and attendance recorder with timestamp.
This project aims at providing a software that can remember human faces along with their names. It can be helpful in applications involving virtual security systems like **CCTV cameras and some survielliance drones in specialised military operations**. It is also helpful in attendance recorder in schools, offices, and any place where presence and time of arrival of people is to be recorded.

#### Adding a new face to the database:
1. New faces can be added to database of the algoritm by placing an image of the person's face into the folder named $\textit{Photos}$.
2. Do ensure that the image has been named by the **name of the person** strictly.
3. The system will automatically extract and store facial features from the provided images for future recognition.

#### Technology Stack:
The Face Recognition System utilizes the following technologies:
1. **OpenCV**: A popular computer vision library that provides essential tools and algorithms for image and video processing.
2. **Face-recognition**: A neural network based on dlib and cmake, specifically designed for face recognition tasks. It effectively recognizes faces from the list of individuals fed into the system.

#### Marking the Presence of Faces:
The presence of a face is marked in $\textit{Presence.csv}$ file along with the time instant when the face is recognised.
The sheet $\textit{Presence.csv}$ gets **cleared off before each new execution** of program.

#### Conclusion:
The Face Recognition System presented here utilizes OpenCV and face-recognition libraries to accurately detect and recognize human faces. By following the instructions outlined in this documentation, users can seamlessly add new faces to the system's database, track the presence of faces, and clear the data sheet for each subsequent execution.
