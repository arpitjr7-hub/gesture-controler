import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- AUDIO SETUP ----------------
devices = AudioUtilities.GetSpeakers()
volume=devices.EndpointVolume
vol_min, vol_max, _ = volume.GetVolumeRange()

# ---------------- MEDIAPIPE SETUP ----------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"  # ensure file exists
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        x1 = int(hand[4].x * frame.shape[1])   # Thumb tip
        y1 = int(hand[4].y * frame.shape[0])

        x2 = int(hand[8].x * frame.shape[1])   # Index tip
        y2 = int(hand[8].y * frame.shape[0])

        dist = np.hypot(x2 - x1, y2 - y1)

        vol = np.interp(dist, [30, 200], [vol_min, vol_max])
        volume.SetMasterVolumeLevel(vol, None)

        # UI
        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cv2.imshow("Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
