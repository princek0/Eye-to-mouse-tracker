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

# Parameters
EAR_THRESHOLD = 0.25  # Threshold for detecting blinks
MIN_FRAMES_FOR_BLINK = 3  # Minimum frames below threshold to count as a blink

# State variables
is_blinking = False
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)  # Detect faces using dlib

    for rect in faces:
        # Draw rectangles around the detected face
        x, y, w, h = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)  # Blue for face

        # Facial landmarks detection
        shape = predictor(gray_frame, rect)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract eye landmarks
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear > 0.2:
            if left_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= MIN_FRAMES_FOR_BLINK and not is_blinking:
                    is_blinking = True
                    print("Left blink detected!")
            
            if right_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= MIN_FRAMES_FOR_BLINK and not is_blinking:
                    is_blinking = True
                    print("Right blink detected!")

        # Draw circles on eye landmarks (optional)
        for (lx, ly) in left_eye + right_eye:
            cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

        # Optional: Detect eyes with Haar cascade (if desired)
        roi_gray = gray_frame[y:h, x:w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)  # Green for eyes

    cv2.imshow('Blink Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
