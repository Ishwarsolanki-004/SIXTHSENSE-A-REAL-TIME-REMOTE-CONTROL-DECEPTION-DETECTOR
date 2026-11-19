from __future__ import annotations

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def _download(url: str, dest_path: str) -> None:
    print(f"Downloading {os.path.basename(dest_path)} ...")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done * 100 // total_size)
        sys.stdout.write(f"\r  {pct:3d}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, _progress)
    print()


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"{filename} already present, skipping.")
            continue
        try:
            _download(url, dest_path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"\nFailed to download {filename}: {exc}")
            print(
                "You can download it manually from:\n  "
                f"{url}\nand place it at:\n  {dest_path}"
            )
            sys.exit(1)
    print("\nAll model files are ready in:", MODELS_DIR)


if __name__ == "__main__":
    main()
