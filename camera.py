from ultralytics import YOLO

model = YOLO("yolo11n.pt")

for result in model.predict(source=0, show=True, stream=True):
    for box in result.boxes:
        cls  = int(box.cls)
        conf = float(box.conf)
        print(f"{model.names[cls]}: {conf:.0%}")