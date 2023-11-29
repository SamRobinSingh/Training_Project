import face_recognition
import cv2
import numpy as np

# Load the images and their corresponding names (students' names)
known_images = ["image.jpg"]
known_names = ["Student 1"]

# Load the known face encodings
known_encodings = []
for image_path in known_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)

# Initialize the webcam or load a video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with the video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Recognize faces in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with the known encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Check if there is a match
        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        # Draw the face rectangle and name on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Smart Attendance System', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()