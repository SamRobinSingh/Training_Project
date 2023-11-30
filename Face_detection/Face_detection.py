import cv2
import dlib

# Load Dlib's face detection model
detector = dlib.get_frontal_face_detector()

# Initialize the webcam or load a video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with the video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Dlib
    faces = detector(gray_frame)

    for face in faces:
        # Get the face bounding box coordinates
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Draw the face bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()