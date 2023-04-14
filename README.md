# Prosence
### Face Recognition and attendance recorder with timestamp.
New faces can be added to database of the algoritm by placing an image of the person's face into the folder named $\textit{Photos}$. We have to ensure that the image has been named by $\textbf{the name of the person}$.

This project uses $\textbf{OpenCV}$ along with $\textbf{face-recognition (a neural network, coded over dlib and cmake)}$ to recognise faces from the list fed to it initially.

The presence of a face is marked in $\textit{Presence.csv}$ file along with the time instant when the face is recognised.
The sheet $\textit{Presence.csv}$ gets $\textbf{cleared off before each new execution}$ of program.
