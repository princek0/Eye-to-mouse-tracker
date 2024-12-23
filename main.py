import cv2
import dlib
from scipy.spatial import distance as dist

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Load the Haar cascade files for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()

    # If the frame was not captured properly, break the loop
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for face

        # Extract the region of interest (ROI) for the detected face
        roi_gray = gray_frame[y:y+h, x:x+w]

        # Detect eyes within the ROI (the face region)
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each detected eye
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)  # Green for eyes

    # Display the frame with rectangles drawn
    cv2.imshow('Face and Eye Detection', frame)

    # Check for a key press to exit the loop (e.g., press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()