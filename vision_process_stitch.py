import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# =========================
# CONFIG
# =========================
VIDEO_PATH = "demo/002.mp4"
TALKNET_PKL = "demo/002/pywork/tracks.pckl"   # your existing pkl
OUTPUT_VIDEO = "output_annotated.mp4"

IOU_THRESHOLD = 0.3
SMOOTHING_WINDOW = 7
SPEAKING_THRESHOLD = 0.6

# =========================
# UTILS
# =========================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

def smooth_scores(scores, window):
    if len(scores) < window:
        return np.mean(scores)
    return np.mean(list(scores)[-window:])

# =========================
# LOAD TALKNET
# =========================
print("[INFO] Loading TalkNet PKL...")
with open(TALKNET_PKL, "rb") as f:
    tracks = pickle.load(f)

# Expected format:
# talknet_data[frame_idx] = [
#   {"bbox": [x1,y1,x2,y2], "score": float}
# ]
    
# Convert tracks list to frame-indexed dictionary
talknet_data = defaultdict(list)
for track in tracks:
    frames = track["track"]["frame"]
    bboxes = track["track"]["bbox"]
    # Assign score 1.0 for detected faces (can be replaced with actual scores if available)
    for frame_idx, bbox in zip(frames, bboxes):
        talknet_data[int(frame_idx)].append({
            "bbox": [int(b) for b in bbox],
            "score": 1.0
        })

# =========================
# YOLO + BYTETRACK
# =========================
print("[INFO] Loading YOLOv8...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# Track speaking scores per ID
speaking_scores = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))

frame_idx = 0

print("[INFO] Processing video...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False
    )

    yolo_tracks = []

    if results and results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box.tolist())
            yolo_tracks.append({
                "id": int(track_id),
                "bbox": [x1, y1, x2, y2]
            })

    # Reset speaking for this frame
    frame_speaking = defaultdict(float)

    # =========================
    # TALKNET ↔ YOLO ASSOCIATION
    # =========================
    faces = talknet_data.get(frame_idx, [])

    for face in faces:
        face_bbox = face["bbox"]
        face_score = face["score"]

        best_iou = 0
        best_id = None

        for track in yolo_tracks:
            overlap = iou(face_bbox, track["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_id = track["id"]

        if best_iou > IOU_THRESHOLD:
            frame_speaking[best_id] = max(frame_speaking[best_id], face_score)

    # =========================
    # UPDATE + DRAW
    # =========================
    for track in yolo_tracks:
        tid = track["id"]
        bbox = track["bbox"]

        speaking_scores[tid].append(frame_speaking.get(tid, 0.0))
        smoothed = smooth_scores(speaking_scores[tid], SMOOTHING_WINDOW)

        speaking = smoothed > SPEAKING_THRESHOLD
        color = (0, 255, 0) if speaking else (255, 0, 0)

        cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
        cv2.putText(
            frame,
            f"ID {tid} | {'SPEAKING' if speaking else 'silent'}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()

print("[DONE] Output saved to:", OUTPUT_VIDEO)
