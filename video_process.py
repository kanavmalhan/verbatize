import cv2
import json
import pickle
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

FPS = 25
IOU_THRESH = 0.3
SPEAKING_THRESH = 0.0  # TalkNet score threshold

# ------------------------
# Utils
# ------------------------

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


# ------------------------
# YOLOv8 + ByteTrack
# ------------------------

def run_bytetrack(video_path, yolo_model="yolov8n-face.pt"):
    model = YOLO(yolo_model)

    tracks = defaultdict(list)

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.3,
        iou=0.5,
        stream=True
    )

    for frame_idx, r in enumerate(results):
        if r.boxes.id is None:
            continue

        for box, tid in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.id.cpu().numpy()):
            tracks[frame_idx].append({
                "track_id": int(tid),
                "bbox": box.tolist()
            })

    return tracks


# ------------------------
# Load TalkNet Outputs
# ------------------------

def load_talknet(pywork_path):
    tracks = pickle.load(open(f"{pywork_path}/tracks.pckl", "rb"))
    scores = pickle.load(open(f"{pywork_path}/scores.pckl", "rb"))
    return tracks, scores


# ------------------------
# Fusion
# ------------------------

def fuse_talknet_bytetrack(talknet_tracks, talknet_scores, bytetrack_tracks):
    frame_events = []

    for t_idx, track in enumerate(talknet_tracks):
        frames = track["track"]["frame"]
        bboxes = track["track"]["bbox"]
        scores = talknet_scores[t_idx]

        for i, frame in enumerate(frames):
            if frame >= len(scores):
                continue

            if scores[i] <= SPEAKING_THRESH:
                continue

            tn_box = bboxes[i]
            candidates = bytetrack_tracks.get(frame, [])

            best_iou, best_id = 0, None
            for c in candidates:
                iou_val = iou(tn_box, c["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = c["track_id"]

            if best_id is not None and best_iou > IOU_THRESH:
                frame_events.append({
                    "frame": frame,
                    "person_id": best_id,
                    "score": float(scores[i])
                })

    return frame_events


# ------------------------
# Build Segments (for Whisper / pyannote)
# ------------------------

def build_segments(frame_events):
    segments = defaultdict(list)

    MIN_CONSEC_FRAMES = 3
    MAX_SILENCE_GAP = 0.3
    MIN_SEGMENT_LEN = 0.3

    by_person = defaultdict(list)
    for e in frame_events:
        by_person[e["person_id"]].append(e)

    for pid, events in by_person.items():
        events.sort(key=lambda x: x["frame"])

        streak = []
        for e in events:
            if not streak:
                streak = [e]
                continue

            if e["frame"] == streak[-1]["frame"] + 1:
                streak.append(e)
            else:
                if len(streak) >= MIN_CONSEC_FRAMES:
                    start = streak[0]["frame"] / FPS
                    end = streak[-1]["frame"] / FPS
                    segments[pid].append({
                        "start": start,
                        "end": end,
                        "confidence": max(s["score"] for s in streak)
                    })
                streak = [e]

        if len(streak) >= MIN_CONSEC_FRAMES:
            start = streak[0]["frame"] / FPS
            end = streak[-1]["frame"] / FPS
            segments[pid].append({
                "start": start,
                "end": end,
                "confidence": max(s["score"] for s in streak)
            })

        # Merge short gaps
        merged = []
        for seg in segments[pid]:
            if not merged:
                merged.append(seg)
                continue

            if seg["start"] - merged[-1]["end"] < MAX_SILENCE_GAP:
                merged[-1]["end"] = seg["end"]
                merged[-1]["confidence"] = max(
                    merged[-1]["confidence"], seg["confidence"]
                )
            else:
                merged.append(seg)

        # Drop tiny segments
        segments[pid] = [
            s for s in merged if (s["end"] - s["start"]) >= MIN_SEGMENT_LEN
        ]

    return [
        {"person_id": pid, "segments": segs}
        for pid, segs in segments.items()
    ]


# ------------------------
# Main
# ------------------------

def main():
    VIDEO_PATH = "demo/002.mp4"
    PYWORK_PATH = "demo/002/pywork"
    OUTPUT_JSON = "speaker_segments.json"

    print("[1] Running YOLOv8 + ByteTrack...")
    bytetrack_tracks = run_bytetrack(VIDEO_PATH)

    print("[2] Loading TalkNet outputs...")
    talknet_tracks, talknet_scores = load_talknet(PYWORK_PATH)

    print("[3] Fusing TalkNet with ByteTrack...")
    frame_events = fuse_talknet_bytetrack(
        talknet_tracks,
        talknet_scores,
        bytetrack_tracks
    )

    print("[4] Building speaker segments...")
    speaker_segments = build_segments(frame_events)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(speaker_segments, f, indent=2)

    print(f"[DONE] Saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
