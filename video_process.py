import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import torch
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
# YOLOv8l + ByteTrack
# ------------------------

def run_bytetrack(video_path, yolo_model="yolov8n-face.pt", device=None):
    """Run YOLO + ByteTrack on a video."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(yolo_model)

    tracks = defaultdict(list)

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.3,
        iou=0.5,
        stream=True,
        device=device,
    )

    for frame_idx, r in enumerate(results):
        if getattr(r.boxes, "id", None) is None:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else r.boxes.xyxy
        ids = r.boxes.id.cpu().numpy() if hasattr(r.boxes.id, "cpu") else r.boxes.id

        for box, tid in zip(xyxy, ids):
            tracks[frame_idx].append({
                "track_id": int(tid),
                "bbox": box.tolist()
            })

    return tracks


# ------------------------
# Load TalkNet Outputs
# ------------------------

def load_talknet(pywork_path):
    with open(os.path.join(pywork_path, "tracks.pckl"), "rb") as track_file:
        tracks = pickle.load(track_file)
    with open(os.path.join(pywork_path, "scores.pckl"), "rb") as score_file:
        scores = pickle.load(score_file)
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse TalkNet outputs with ByteTrack detections."
    )
    parser.add_argument(
        "--videoName",
        type=str,
        required=True,
        help="Video name without extension, for example 003",
    )
    parser.add_argument(
        "--videoFolder",
        type=str,
        default="demo",
        help="Base folder containing the source videos and TalkNet outputs.",
    )
    parser.add_argument(
        "--videoPath",
        type=str,
        default=None,
        help="Optional explicit path to the source video file.",
    )
    parser.add_argument(
        "--pyworkPath",
        type=str,
        default=None,
        help="Optional explicit path to the TalkNet pywork directory.",
    )
    parser.add_argument(
        "--outputJson",
        type=str,
        default=None,
        help="Optional explicit path for the speaker segments JSON.",
    )
    parser.add_argument(
        "--yoloModel",
        type=str,
        default="yolov8n-face.pt",
        help="YOLO face model path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device override, for example cuda or cpu.",
    )
    return parser.parse_args()


def resolve_defaults(args):
    video_root = Path(args.videoFolder)
    save_root = video_root / args.videoName

    video_path = Path(args.videoPath) if args.videoPath else video_root / f"{args.videoName}.mp4"
    pywork_path = Path(args.pyworkPath) if args.pyworkPath else save_root / "pywork"
    output_json = Path(args.outputJson) if args.outputJson else save_root / "speaker_segments.json"

    return video_path, pywork_path, output_json

def main():
    args = parse_args()
    VIDEO_PATH, PYWORK_PATH, OUTPUT_JSON = resolve_defaults(args)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not PYWORK_PATH.exists():
        raise FileNotFoundError(f"TalkNet output folder not found: {PYWORK_PATH}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("[1] Running YOLO + ByteTrack...")
    bytetrack_tracks = run_bytetrack(
        str(VIDEO_PATH),
        yolo_model=args.yoloModel,
        device=args.device,
    )

    print("[2] Loading TalkNet outputs...")
    talknet_tracks, talknet_scores = load_talknet(str(PYWORK_PATH))

    print("[3] Fusing TalkNet with ByteTrack...")
    frame_events = fuse_talknet_bytetrack(
        talknet_tracks,
        talknet_scores,
        bytetrack_tracks
    )

    print("[4] Building speaker segments...")
    speaker_segments = build_segments(frame_events)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as output_file:
        json.dump(speaker_segments, output_file, indent=2)

    print(f"[DONE] Saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
