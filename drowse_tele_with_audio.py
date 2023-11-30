import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import telebot
import auth
from pydub import AudioSegment
from pydub.playback import play


#Konstanta untuk rasio aspek mata (EAR) dan ambang kantuk
EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 20

# Token API bot Telegram dan ID obrolan
TOKEN = auth.tele_token
CHAT_ID = auth.tele_chat_id

# Inisialisasi bot Telegram
bot = telebot.TeleBot(TOKEN)

# Berfungsi untuk menghitung rasio aspek mata (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Muat detektor wajah dan prediktor landmark dari dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Mulai merekam video
video_capture = cv2.VideoCapture(0)

# Inisialisasi penghitung bingkai dan tanda kantuk
frame_counter = 0
drowsy = False
lastDrowsy = drowsy

song = AudioSegment.from_mp3("sound/Awas.mp3")

while True:
    # Baca bingkai dari aliran video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Ubah bingkai menjadi skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam bingkai skala abu-abu
    faces = detector(gray, 0)

    for face in faces:
        # Tentukan landmark wajah untuk daerah wajah
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape) #data landmark wajah
        print(len(shape))
        print(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # Extract the coordinates of the left and right eye
        left_eye = shape[42:48] #mengambil data landmark index 42-47 untuk mata kiri
        right_eye = shape[36:42] #mengambil data landmark index 36-41 untuk mata kanan

        # Hitung rasio aspek mata (EAR) untuk kedua mata
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Rata-rata TELINGA untuk kedua mata
        ear = (left_ear + right_ear) / 2.0

        # Periksa apakah rasio aspek mata berada di bawah ambang kantuk
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= EAR_CONSEC_FRAMES:
                
               
                play(song)
                
                drowsy = True
                cv2.putText(frame, "Peringatan! Kantuk Terdeteksi", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Ambil gambar dan simpan
                image_name = "drowsiness_capture.jpg"
                cv2.imwrite(image_name, frame)

                # Kirim pemberitahuan dengan gambar ke Telegram
                if drowsy and not lastDrowsy:
                    with open(image_name, "rb") as photo:
                        bot.send_photo(CHAT_ID, photo)
                        pesan = "EAR = " + str(round(ear,2)) + "\n\nKiri = " + str(round(left_ear,2)) + "\nKanan = " + str(round(right_ear,2)) 
                        bot.send_message(CHAT_ID, pesan)

        else:
            frame_counter = 0
            drowsy = False

           

        lastDrowsy = drowsy

        # Menampilkan rasio aspek mata (EAR) yang dihitung pada bingkai
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Gambarlah persegi panjang daerah mata pada bingka
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.rectangle(frame, (face.left(), face.top() - 35), (face.right(), face.top()), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Face", (face.left() + 6, face.top() - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tampilkan bingkai yang dihasilkan
    cv2.imshow("Drowsiness Detection", frame)

    # Putuskan loop jika 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan objek pengambilan video dan tutup semua jendela
video_capture.release()
cv2.destroyAllWindows()
