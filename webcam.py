import cv2
import numpy as np
from collections import deque
from huggingface_hub import hf_hub_download
from tensorflow import keras
from ultralytics import YOLO

classes = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
names = ["Surprise", "Peur", "Degout", "Joie", "Tristesse", "Colere", "Neutre"]
colors = [
    (0, 140, 255),    # surprise : orange
    (200, 0, 150),    # peur : violet
    (0, 170, 0),      # degout : vert
    (0, 220, 255),    # joie : jaune
    (255, 100, 0),    # tristesse : bleu
    (0, 0, 255),      # colere : rouge
    (150, 150, 150),  # neutre : gris
]

model = keras.models.load_model("expressions_cnn.keras")
detector = YOLO(hf_hub_download("arnabdhar/YOLOv8-Face-Detection", "model.pt"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("expressions", cv2.WINDOW_AUTOSIZE)
tracks = {}
next_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = detector(frame, conf=0.4, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    used = set()
    crops = []
    clean = frame.copy()
    biggest = (0, None)

    for box in boxes:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        # on rattache la boite au visage suivi qui a le meilleur IoU
        best, best_iou = None, 0.3
        for tid, t in tracks.items():
            if tid in used:
                continue
            v = iou(box, t["box"])
            if v > best_iou:
                best, best_iou = tid, v
        if best is None:
            best = next_id
            next_id += 1
            tracks[best] = {"probs": deque(maxlen=10)}
        used.add(best)
        tracks[best]["box"] = box
        tracks[best]["missed"] = 0

        # meme pretraitement YOLO, RGB, 100x100, /255
        face = cv2.cvtColor(clean[max(y1, 0):y2, max(x1, 0):x2], cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (100, 100)).astype("float32") / 255
        crops.append(face)

        p = model(face[None], training=False).numpy()[0]

        tracks[best]["probs"].append(p)
        mean = np.mean(tracks[best]["probs"], axis=0)
        k = mean.argmax()

        if (x2 - x1) * (y2 - y1) > biggest[0]:
            biggest = ((x2 - x1) * (y2 - y1), mean)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[k], 2)
        cv2.putText(frame, f"#{best} {names[k]} {mean[k]:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[k], 2)

    for tid in list(tracks):
        if tid not in used:
            tracks[tid]["missed"] += 1
            if tracks[tid]["missed"] > 10:
                del tracks[tid]

    # ce que voit le CNN (visages en haut a gauche)
    for i, c in enumerate(crops[:3]):
        small = cv2.resize((c * 255).astype("uint8"), (96, 96))
        frame[10:106, 10 + i * 106:106 + i * 106] = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    scale = 2000 / frame.shape[1]
    out = cv2.resize(frame, None, fx=scale, fy=scale)

    # probabilites du visage le plus grand (en haut a droite)
    if biggest[1] is not None:
        W = out.shape[1]
        pw, row = int(W * 0.22), int(W * 0.022)
        x0, y0 = W - pw - 10, 10
        for i, v in enumerate(biggest[1]):
            y = y0 + i * row
            bx = x0 + int(pw * 0.38)
            cv2.rectangle(out, (bx, y + 4), (bx + int(pw * 0.42 * v), y + row - 4), colors[i], -1)
            for text, tx in [(names[i], x0), (f"{v:.2f}", bx + int(pw * 0.45))]:
                cv2.putText(out, text, (tx, y + row - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            row / 40, (0, 0, 0), 40)
                cv2.putText(out, text, (tx, y + row - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            row / 40, (255, 255, 255), 2)

    cv2.imshow("expressions", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.getWindowProperty("expressions", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
