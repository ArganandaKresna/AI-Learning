import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Loading data dari file pickle
model_dict = pickle.load(open('C:/Users/VICTUS/Documents/MachineLearningProject/model.p', 'rb'))
model = model_dict['model']

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

# Inisialisasi Library Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Define label buat Huruf a-z
labels_dict = {i: chr(65 + i) for i in range(26)}

# Variabel buat tracking waktu dan apa alfabet yang muncul
detected_char = None
char_start_time = None
recognized_text = ""
char_duration_threshold = 2.5  # seconds

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Gagal Mendapat Gambar.")
        break

    H, W, _ = frame.shape

    # Convert gambar ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ngeproses gambar pake Mediapipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # Dibuat biar bisa ngeproses 2 tangan
            # Gambarin landmark tangan
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Nge ekstrak landmark tangannya
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize landmark coordinates
            x_min, y_min = min(x_), min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - x_min)
                data_aux.append(landmark.y - y_min)

        ''' 
        Disini sempat ada masalah dimana algoritma Random Forest Classifier harus mendapat 84 feature yang mana
        artinya 2 tangan harus ada dilayar, jadi ditambahkan padding agar program tidak crash akibat Random Forest yang
        ada tidak stop bekerja
        '''
        if len(results.multi_hand_landmarks) == 1:
            data_aux += [0] * (84 - len(data_aux))

        # Kalkulasi koordinat bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        try:
            # Ngebuat prediksi karakter
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Logika if else buat ngecek karakternya ada dalam batas 2.5 detik
            if detected_char == predicted_character:
                if char_start_time is None:
                    char_start_time = time.time()
                elif time.time() - char_start_time > char_duration_threshold:
                    if recognized_text == "" or recognized_text[-1] != predicted_character:
                        recognized_text += predicted_character
                        char_start_time = None
            else:
                detected_char = predicted_character
                char_start_time = time.time()

            # Gambarin bounding box di layar
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Prediksi Error: {e}")

    # Ngedisplay karakter yang kedetek
    cv2.putText(frame, f"Bahasa Isyarat: {recognized_text}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Ngedisplay gambar
    cv2.imshow('Deteksi ASL', frame)

    # Close program kalo tombol "Q" di keyboard diteken
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()