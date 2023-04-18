from datetime import datetime
import face_recognition
import numpy as np
import cv2 as cv
import os
import time

#Directory path for training images
path = 'Photos'
list = os.listdir(path)
images = []
names = []

#Array of images and names
for cls in list:
    img = cv.imread(f'{path}/{cls}')
    images.append(img)
    names.append(os.path.splitext(cls)[0])
    #[0] to remove extensions from names

#Encoding images in array
def encodings(images):
    enclist = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        enclist.append(encode)
    return enclist

KnownList = encodings(images)

#Mark presence in CSV file
def presence(name):
    with open('Presence.csv','r+') as f:
        data = f.readlines()
        nameList = []
        for line in data:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')

#Get input from camera
cap = cv.VideoCapture(0)

t1=time.time()
t=0

while True:
    t+=1
    success, img = cap.read()
    
    #Reduce size to 1/4th to speed up
    imgs = cv.resize(img,(0,0),None,0.25,0.25)
    imgs = cv.cvtColor(imgs,cv.COLOR_BGR2RGB)
    face_frame = face_recognition.face_locations(imgs)
    enc_frame = face_recognition.face_encodings(imgs,face_frame)
    
    #Compare faces
    for encode_face, face_loc in zip(enc_frame, face_frame):
        matches = face_recognition.compare_faces(KnownList, encode_face)
        face_dist = face_recognition.face_distance(KnownList, encode_face)
        #face_dist is array of distances of input from known images. Min needed.
        
        match_index = np.argmin(face_dist)
        if matches[match_index]:
            name = names[match_index]
            top,right,bottom,left = face_loc
            top,right,bottom,left = top*4,right*4,bottom*4,left*4
            cv.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
            cv.rectangle(img,(left,top),(right,top-35),(0,255,0),cv.FILLED)
            cv.putText(img,name,(left+10,top-10),cv.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
            presence(name)
    
    cv.imshow('Detections',img)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break

t2=time.time()
cap.release()
cv.destroyAllWindows()
print(t/(t2-t1),"frames analysed per sec")