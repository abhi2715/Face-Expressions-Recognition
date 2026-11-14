"""Real-time facial expression detection from the webcam (simple OpenCV window).

Needs models/ (see README).

Usage:  venv/bin/python realtimedetection.py [--camera 0]
Press 'q' or ESC to quit.
"""
import argparse
import sys

import cv2

print("Loading models...", flush=True)
from app.engine import EmotionEngine  # noqa: E402

COLORS = {"angry": (82, 82, 240), "disgust": (22, 204, 132), "fear": (250, 139, 167), "happy": (21, 204, 250),
          "neutral": (184, 163, 148), "sad": (250, 165, 96), "surprise": (60, 146, 251)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    engine = EmotionEngine()
    tracker = engine.new_tracker()
    print("Model loaded. Opening camera...", flush=True)

    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        sys.exit(
            f"Cannot open camera {args.camera}. On macOS, grant camera access to your terminal app in "
            "System Settings > Privacy & Security > Camera, then restart the terminal."
        )

    try:
        while True:
            ok, frame = webcam.read()
            if not ok:
                print("Failed to read frame from camera.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            for face in engine.analyze(frame, tracker):
                bx, by, bw, bh = face["box"]
                x, y, fw, fh = int(bx * w), int(by * h), int(bw * w), int(bh * h)
                color = COLORS[face["emotion"]]
                text = f"{face['emotion']} {face['confidence'] * 100:.0f}%"
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                ty = y - 10 if y - 10 > th else y + fh + th + 10
                cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 6, ty + 4), color, -1)
                cv2.putText(frame, text, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

            cv2.imshow("Facial Expression Recognition (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
