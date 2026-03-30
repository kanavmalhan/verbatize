import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full video pipeline in order for a given video name."
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
        help="Base folder containing the source video files.",
    )
    return parser.parse_args()


def run_step(script_name, video_name, video_folder):
    command = [
        sys.executable,
        script_name,
        "--videoName",
        video_name,
        "--videoFolder",
        video_folder,
    ]
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    video_path = Path(args.videoFolder) / f"{args.videoName}.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    subprocess.run([sys.executable, "demo.py", "--videoName", args.videoName], check=True)
    subprocess.run([sys.executable, "video_process.py", "--videoName", args.videoName], check=True)
    subprocess.run([sys.executable, "transcribe.py", "--videoName", args.videoName], check=True)

    print(f"[DONE] Pipeline complete for videoName={args.videoName}")


if __name__ == "__main__":
    main()
