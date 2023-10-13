import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import telebot
import auth
from pydub import AudioSegment
from pydub.playback import play


# Constants for eye aspect ratio (EAR) and drowsiness thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Telegram bot API token and chat ID
TOKEN = auth.tele_token
CHAT_ID = auth.tele_chat_id

# Initialize Telegram bot
bot = telebot.TeleBot(TOKEN)

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start capturing video
video_capture = cv2.VideoCapture(0)

# Initialize frame counters and drowsiness flag
frame_counter = 0
drowsy = False
lastDrowsy = drowsy

song = AudioSegment.from_mp3("sound/among.mp3")

while True:
    # Read frame from video stream
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the coordinates of the left and right eye
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate the eye aspect ratio (EAR) for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if the eye aspect ratio is below the drowsiness threshold
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= EAR_CONSEC_FRAMES:
                
                               
                play(song)
                
                drowsy = True
                cv2.putText(frame, "Peringatan! Kantuk Terdeteksi", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Capture image and save it
                image_name = "drowsiness_capture.jpg"
                cv2.imwrite(image_name, frame)

                # Send notification with image to Telegram
                with open(image_name, "rb") as photo:
                    bot.send_photo(CHAT_ID, photo)

        else:
            frame_counter = 0
            drowsy = False

        lastDrowsy = drowsy

        # Display the calculated eye aspect ratio (EAR) on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw the eye region rectangles on the frame
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.rectangle(frame, (face.left(), face.top() - 35), (face.right(), face.top()), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Face", (face.left() + 6, face.top() - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
