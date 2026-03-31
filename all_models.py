import torch
import cv2
import time
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import os

# -------------------- AUDIO --------------------
def speak(text):
    filename = "temp.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# -------------------- LOAD MODELS --------------------
model_v5 = torch.hub.load('ultralytics/yolov5', 'custom', path='zebra_crossing.pt')
model_v5.conf = 0.25

model_v8 = YOLO("yolov8n.pt")  # general objects
traffic_model = YOLO("best_traffic_nano_yolo.pt")  # traffic light colors

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

last_spoken_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------- RUN ALL MODELS ON SAME FRAME --------------------
    results_v5 = model_v5(frame)
    results_v8 = model_v8(frame, verbose=False)
    results_traffic = traffic_model(frame, verbose=False)

    # -------------------- DRAW OUTPUT --------------------
    frame_v5 = results_v5.ims[0]
    frame_v8 = results_v8[0].plot()
    frame_traffic = results_traffic[0].plot()

    combined = cv2.hconcat([frame_v5, frame_v8, frame_traffic])
    cv2.imshow("YOLOv5 | YOLOv8 | Traffic", combined)

    # -------------------- COLLECT ALL DETECTIONS --------------------
    current_detected = set()

    # YOLOv5
    for *box, conf, cls in results_v5.xyxy[0]:
        current_detected.add(model_v5.names[int(cls)])

    # YOLOv8
    for box in results_v8[0].boxes:
        current_detected.add(model_v8.names[int(box.cls)])

    # Traffic light model
    for box in results_traffic[0].boxes:
        current_detected.add(traffic_model.names[int(box.cls)])

    # -------------------- AUDIO EVERY 10 SEC --------------------
    current_time = time.time()

    if current_time - last_spoken_time >= 10:
        if current_detected:
            text = "Detected " + ", ".join(current_detected)
        else:
            text = "No objects detected"

        print(text)
        speak(text)

        last_spoken_time = current_time

    # -------------------- EXIT --------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()