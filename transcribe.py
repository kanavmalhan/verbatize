import argparse
import json
import math
from pathlib import Path

from faster_whisper import WhisperModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio and align it to fused speaker segments."
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
        help="Base folder containing the processed video outputs.",
    )
    parser.add_argument(
        "--segmentsJson",
        type=str,
        default=None,
        help="Optional explicit path to the speaker segments JSON.",
    )
    parser.add_argument(
        "--audioFile",
        type=str,
        default=None,
        help="Optional explicit path to the extracted audio wav file.",
    )
    parser.add_argument(
        "--outputTxt",
        type=str,
        default=None,
        help="Optional explicit path for the final transcript text file.",
    )
    parser.add_argument(
        "--whisperModel",
        type=str,
        default="large-v2",
        help="Faster-Whisper model size or path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Whisper inference device.",
    )
    parser.add_argument(
        "--computeType",
        type=str,
        default="float16",
        help="Whisper compute type.",
    )
    return parser.parse_args()


def resolve_defaults(args):
    save_root = Path(args.videoFolder) / args.videoName
    segments_json = Path(args.segmentsJson) if args.segmentsJson else save_root / "speaker_segments.json"
    audio_file = Path(args.audioFile) if args.audioFile else save_root / "pyavi" / "audio.wav"
    output_txt = Path(args.outputTxt) if args.outputTxt else save_root / "final_transcript.txt"
    return segments_json, audio_file, output_txt


def load_speaker_segments(segments_json):
    with open(segments_json, "r", encoding="utf-8") as segment_file:
        speaker_data = json.load(segment_file)

    speaker_segments = []
    for person in speaker_data:
        for seg in person["segments"]:
            speaker_segments.append(
                {
                    "person_id": person["person_id"],
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )
    return speaker_segments


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(w_start, w_end, speaker_segments):
    best_overlap = 0.0
    best_speaker = None

    for seg in speaker_segments:
        current_overlap = overlap(w_start, w_end, seg["start"], seg["end"])
        if current_overlap > best_overlap:
            best_overlap = current_overlap
            best_speaker = seg["person_id"]

    if best_speaker is not None:
        return f"Person {best_speaker}"

    min_dist = math.inf
    closest = None
    for seg in speaker_segments:
        dist = min(abs(w_start - seg["end"]), abs(w_end - seg["start"]))
        if dist < min_dist:
            min_dist = dist
            closest = seg["person_id"]

    return f"Person {closest}" if closest is not None else "Unknown"


def build_transcript_lines(whisper_segments, speaker_segments):
    lines = []

    for seg in whisper_segments:
        speaker = assign_speaker(seg.start, seg.end, speaker_segments)
        text = seg.text.strip()

        if not text:
            continue

        lines.append(f"[{seg.start:06.2f}-{seg.end:06.2f}] {speaker}: {text}")

    return lines


def main():
    args = parse_args()
    segments_json, audio_file, output_txt = resolve_defaults(args)

    if not segments_json.exists():
        raise FileNotFoundError(f"Speaker segments not found: {segments_json}")
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    speaker_segments = load_speaker_segments(segments_json)

    print(f"[INFO] Loading Whisper model: {args.whisperModel}")
    model = WhisperModel(
        args.whisperModel,
        device=args.device,
        compute_type=args.computeType,
    )

    print("[INFO] Transcribing audio...")
    whisper_segments, _ = model.transcribe(
        str(audio_file),
        vad_filter=True,
        beam_size=5,
    )

    lines = build_transcript_lines(whisper_segments, speaker_segments)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))

    print("[DONE] Transcript saved to", output_txt)


if __name__ == "__main__":
    main()
    raise SystemExit
"""

# -------------------------
# Speaker assignment logic
# -------------------------
def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def assign_speaker(w_start, w_end):
    best_overlap = 0.0
    best_speaker = None

    for seg in speaker_segments:
        o = overlap(w_start, w_end, seg["start"], seg["end"])
        if o > best_overlap:
            best_overlap = o
            best_speaker = seg["person_id"]

    if best_speaker is not None:
        return f"Person {best_speaker}"

    # fallback: closest in time
    min_dist = math.inf
    closest = None
    for seg in speaker_segments:
        dist = min(abs(w_start - seg["end"]), abs(w_end - seg["start"]))
        if dist < min_dist:
            min_dist = dist
            closest = seg["person_id"]

    return f"Person {closest}"

# -------------------------
# Build transcript
# -------------------------
lines = []

for seg in whisper_segments:
    speaker = assign_speaker(seg.start, seg.end)
    text = seg.text.strip()

    if not text:
        continue

    lines.append(
        f"[{seg.start:06.2f}–{seg.end:06.2f}] {speaker}: {text}"
    )

# -------------------------
# Save output
# -------------------------
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("[DONE] Transcript saved to", OUTPUT_TXT)
"""
