import cv2
import numpy as np

# Function to perform color-based object tracking
def track_object_by_color(lower_color, upper_color):
    cap = cv2.VideoCapture(0)  # Use default camera 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a binary mask to filter the object based on the color range
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        # Find the contours of the object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box around the tracked object
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Change this value based on the size of your object
                # Calculate the mean color of the contour
                mean_color = cv2.mean(hsv_frame, mask=mask)[0:3]
                if lower_color[0] <= mean_color[0] <= upper_color[0] and \
                   lower_color[1] <= mean_color[1] <= upper_color[1] and \
                   lower_color[2] <= mean_color[2] <= upper_color[2]:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (214, 255, 51), 2)  # Color (214, 255, 51) for #33FFD6

        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define the color range for the object you want to track 
    lower_color = np.array([85, 127, 127], dtype=np.uint8)  
    upper_color = np.array([100, 255, 255], dtype=np.uint8)  

    track_object_by_color(lower_color, upper_color)
