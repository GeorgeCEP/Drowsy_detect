import cv2
import dlib
from scipy.spatial import distance

# Initialize the face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# EAR thresholds (you'll need to calibrate these based on your own face)
EAR_MIN = 0.18  
EAR_MAX = 0.35  

# Function to calculate EAR
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Normalize func
def calculate_drowsiness_level(ear):
    if ear > EAR_MAX:
        ear = EAR_MAX
    elif ear < EAR_MIN:
        ear = EAR_MIN

    # Calculate drowsiness level (0: fully open, 1: fully closed or squinted)
    drowsiness_level = (EAR_MAX - ear) / (EAR_MAX - EAR_MIN)
    
    return drowsiness_level

# Process func
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract left and right eye coordinates
        left_eye_pts = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_pts = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear = (left_ear + right_ear) / 2.0


        print(f"EAR: {ear:.2f}")

        # Calculate drowsiness level
        drowsiness_level = calculate_drowsiness_level(ear)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Place the EAR and drowsiness score near the face
        cv2.putText(frame, f"Drowsiness: {drowsiness_level:.2f}", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame

# Function to start the webcam and process the video stream
def process_webcam():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = process_frame(frame)
        cv2.imshow("Drowsiness Detection", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()
