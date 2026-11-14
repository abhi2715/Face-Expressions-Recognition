"""Re-crop every dataset face the way the live app does (YuNet detect -> eye-level -> FER framing).

Training on these teaches the model the exact framing it sees from the camera.

Usage:  venv/bin/python prepare_aligned.py
Output: cache/aligned_train.npz, cache/aligned_test.npz  (same order as train_torch.load_dataset)
"""
import os

import cv2
import numpy as np

from app.engine import FaceDetector, align_crop
from train_torch import TEST_DIR, TRAIN_DIR, load_dataset

CACHE_DIR = "cache"


def realign(images, detector):
    out = images.copy()
    found = np.zeros(len(images), dtype=bool)
    for i, img in enumerate(images):
        big = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 120, 120, 200, 200, cv2.BORDER_REPLICATE)
        faces = detector.detect(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
        if faces:
            out[i] = align_crop(big, max(faces, key=lambda f: f[2] * f[3]))
            found[i] = True
        if i % 5000 == 0:
            print(f"  {i}/{len(images)}", flush=True)
    return out, found


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    detector = FaceDetector(score_threshold=0.5)
    for name, directory in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        images, targets = load_dataset(directory)
        aligned, found = realign(images, detector)
        np.savez_compressed(os.path.join(CACHE_DIR, f"aligned_{name}.npz"), images=aligned, found=found, targets=targets)
        print(f"{name}: {found.mean():.1%} faces re-detected", flush=True)


if __name__ == "__main__":
    main()
