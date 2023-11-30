import cv2
import face_recognition

person_image_path = "image.jpg"
person_image = face_recognition.load_image_file(person_image_path)
person_encoding = face_recognition.face_encodings(person_image)[0]

# Function to perform face verification using Haar cascades and face recognition
def verify_person_in_live_video(person_encoding):
    cap = cv2.VideoCapture(0)  # Use default camera (change to a specific camera index if needed)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB color space (required for face_recognition)
        rgb_frame = frame[:, :, ::-1]

        # Find faces in the frame using Haar cascades
        face_locations = face_recognition.face_locations(rgb_frame)

        # If a face is found, try to recognize it
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for face_encoding in face_encodings:
                # Compare the face with the known person's image
                result = face_recognition.compare_faces([person_encoding], face_encoding)
                if result[0]:
                    print("Face matched!")
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) 
                else:
                    print("Face not matched!")

        cv2.imshow('Face Verification', frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_person_in_live_video(person_encoding)