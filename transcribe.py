import json
import math
from faster_whisper import WhisperModel

SEGMENTS_JSON = "speaker_segments.json"
AUDIO_FILE = "demo/002/pyavi/audio.wav"
OUTPUT_TXT = "final_transcript.txt"

WHISPER_MODEL = "large-v2"  # much better accuracy, ~3.5GB VRAM on GPU
DEVICE = "cuda"
COMPUTE_TYPE = "float16"    # faster on T4, lower VRAM than float32

# -------------------------
# Load speaker segments
# -------------------------
with open(SEGMENTS_JSON, "r") as f:
    speaker_data = json.load(f)

speaker_segments = []
for person in speaker_data:
    for seg in person["segments"]:
        speaker_segments.append({
            "person_id": person["person_id"],
            "start": seg["start"],
            "end": seg["end"]
        })

# -------------------------
# Load Whisper
# -------------------------
print("[INFO] Loading Whisper large-v2 model...")
model = WhisperModel(
    WHISPER_MODEL,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("[INFO] Transcribing audio...")
whisper_segments, _ = model.transcribe(
    AUDIO_FILE,
    vad_filter=True,
    beam_size=5
)

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