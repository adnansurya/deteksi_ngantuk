import cv2
import dlib
from scipy.spatial import distance

def calculate_ear(eye):
    # Vertical eye landmarks (indexes: 1, 2, 5, 4)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Horizontal eye landmarks (indexes: 0, 3)
    C = distance.euclidean(eye[0], eye[3])

    # Eye aspect ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness():
    # Load the facial landmarks predictor from dlib
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Load the face cascade file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Initialize drowsiness counter and threshold
    drowsy_count = 0
    drowsy_threshold = 15

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Get the face landmarks using dlib
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # Extract the eye regions
            left_eye = shape[42:48]
            right_eye = shape[36:42]

            # Calculate the eye aspect ratio (EAR) for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            # Average the EAR for both eyes
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw the eyes and annotate EAR on the frame
            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
            cv2.putText(frame, "EAR: {:.2f}".format(avg_ear), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check if the average eye aspect ratio is below the threshold
            if avg_ear < drowsy_threshold:
                drowsy_count += 1
                if drowsy_count >= drowsy_threshold:
                    cv2.putText(frame, "Drowsiness Detected!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                drowsy_count = 0

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_drowsiness()
