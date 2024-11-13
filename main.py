import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load training images and their names
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

print("Training images:", myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:  # Ensure the image is loaded correctly
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print("Class names:", classNames)

# Function to align the face based on the position of the eyes
def align_face(img, face_landmarks):
    if "left_eye" in face_landmarks and "right_eye" in face_landmarks:
        left_eye = face_landmarks["left_eye"]
        right_eye = face_landmarks["right_eye"]

        # Calculate the center of both eyes
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")

        # Calculate the angle between the eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate the center of the eyes for rotation
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                       (left_eye_center[1] + right_eye_center[1]) / 2)

        # Create the rotation matrix around the eyes' center
        M = cv2.getRotationMatrix2D((float(eyes_center[0]), float(eyes_center[1])), angle, scale=1)

        # Apply affine transformation to align the face
        aligned_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return aligned_face
    return img


# Function to find encodings for training images
def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        if face_locations:
            # Only process the first face for training images
            face_landmarks = face_recognition.face_landmarks(img_rgb, face_locations)[0]
            aligned_face = align_face(img_rgb, face_landmarks)
            encode = face_recognition.face_encodings(aligned_face, known_face_locations=[face_locations[0]])[0]
            encodeList.append(encode)
    return encodeList

# Function to mark attendance in a CSV file
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:  # Only mark attendance if not already marked
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Generate encodings for known images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:  # Check if the frame is captured successfully
        print("Failed to capture image from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize image for faster processing
    imgS_rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS_rgb)
    encodesCurFrame = []

    for faceLoc in facesCurFrame:
        face_landmarks = face_recognition.face_landmarks(imgS_rgb, [faceLoc])[0]
        aligned_face = align_face(imgS_rgb, face_landmarks)
        encodeFace = face_recognition.face_encodings(aligned_face, known_face_locations=[faceLoc])[0]
        encodesCurFrame.append((encodeFace, faceLoc))

    for encodeFace, faceLoc in encodesCurFrame:
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()
        else:
            name = "Unknown Person"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale up face location

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        if name != "Unknown Person":
            markAttendance(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
