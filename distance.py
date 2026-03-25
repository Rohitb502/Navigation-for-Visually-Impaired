import cv2
import torch
import numpy as np
from ultralytics import YOLO

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transforms   = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

model = YOLO("yolo11n.pt")   

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    yolo_results = model(frame, verbose=False)[0]

    img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)

    with torch.no_grad():
        depth_pred = midas(input_batch)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = depth_pred.cpu().numpy()

    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    output = frame.copy()

    for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls             = int(box.cls)
        conf            = float(box.conf)
        label           = model.names[cls]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        raw_depth  = depth_map[cy, cx]

        depth_min  = depth_map.min()
        depth_max  = depth_map.max()
        rel_depth  = 1.0 - (raw_depth - depth_min) / (depth_max - depth_min + 1e-6)
        rel_meters = rel_depth * 10  

        color = (0, 255, 0) if rel_meters > 5 else (0, 0, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%} | {rel_meters:.1f}m"
        cv2.putText(output, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.circle(output, (cx, cy), 4, (255, 255, 0), -1)

    depth_bgr   = depth_colored          # already BGR
    combined    = np.hstack([output, depth_bgr])

    cv2.imshow("YOLO + MiDaS Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
