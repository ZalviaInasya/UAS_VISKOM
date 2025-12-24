import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===== LOAD MODEL =====
model = load_model("emotion_model.h5")

# Label emosi (SESUIKAN URUTAN TRAINING)
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ===== LOAD FACE DETECTOR =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===== WARNA KOTAK (HEX #EE6983 -> BGR) =====
BOX_COLOR = (131, 105, 238)  # BGR

# ===== BUKA KAMERA =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, frame = cap.read()
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        preds = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # ===== DRAW BOX =====
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            BOX_COLOR,
            2
        )

        # ===== TEXT =====
        text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            BOX_COLOR,
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
