"""Build the FER+ dataset cache (Microsoft's 10-annotator relabelling of FER-2013).

Inputs (in data/ferplus_raw/):
  train.parquet / valid.parquet / test.parquet  FER-2013 in original fer2013.csv row order
                                                (Training / PublicTest / PrivateTest), from HF AutumnQiu/fer2013
  fer2013new.csv                                FER+ vote counts, github.com/microsoft/FERPlus

Output: cache/ferplus.npz with
  images  (N,48,48) uint8, votes (N,7) float32 in LABELS order, split (N,) 0=train 1=val 2=test

Labels follow the FER+ paper's "majority" protocol: per-image votes of 1 are treated as noise and dropped,
images whose top vote is unknown / not-a-face / contempt are removed. Contempt is dropped so the apps
keep their 7 emotions.

Usage: venv/bin/python prepare_ferplus.py
"""
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

RAW = "data/ferplus_raw"
OUT = "cache/ferplus.npz"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# fer2013new.csv column -> our label
FERPLUS_COLS = {"anger": "angry", "disgust": "disgust", "fear": "fear", "happiness": "happy",
                "neutral": "neutral", "sadness": "sad", "surprise": "surprise"}
# fer2013.csv emotion codes (also the parquet's)
FER_CODES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_images():
    images, fer_labels, usage = [], [], []
    for split, name in [("train", "Training"), ("valid", "PublicTest"), ("test", "PrivateTest")]:
        df = pd.read_parquet(os.path.join(RAW, f"{split}.parquet"))
        for img, label in zip(df["image"], df["label"]):
            arr = np.array(Image.open(io.BytesIO(img["bytes"])).convert("L"))
            assert arr.shape == (48, 48), arr.shape
            images.append(arr)
            fer_labels.append(FER_CODES[int(label)])
            usage.append(name)
    return np.stack(images), fer_labels, usage


def check_against_local(images, fer_labels, usage):
    """The local Kaggle copy names files <Usage>_<id>.jpg under its FER-2013 label: confirm row order matches."""
    local = {}
    for top in ["images/train", "images/test"]:
        for label in LABELS:
            d = os.path.join(top, label)
            for f in os.listdir(d):
                if f.endswith(".jpg"):
                    arr = np.array(Image.open(os.path.join(d, f)).convert("L"))
                    key = (arr.astype(np.int16)[::4, ::4] // 32).tobytes()  # tolerant to JPEG re-encoding
                    local.setdefault(key, []).append((f.split("_")[0], label))
    matched = agree = 0
    for img, lab, use in zip(images, fer_labels, usage):
        hits = local.get((img.astype(np.int16)[::4, ::4] // 32).tobytes())
        if hits:
            matched += 1
            agree += any(u == use and l == lab for u, l in hits)
    print(f"local cross-check: {matched}/{len(images)} images found locally, "
          f"{agree}/{matched} agree on split+label")
    return agree / max(matched, 1)


def main():
    images, fer_labels, usage = load_images()
    plus = pd.read_csv(os.path.join(RAW, "fer2013new.csv"))
    assert len(plus) == len(images), (len(plus), len(images))
    assert list(plus["Usage"]) == usage, "FER+ Usage column does not line up with image order"

    # The fer2013.csv labels must broadly agree with FER+ majority labels if the row order is right
    top = plus[list(FERPLUS_COLS)].to_numpy().argmax(1)
    top_names = [FERPLUS_COLS[list(FERPLUS_COLS)[i]] for i in top]
    print(f"FER-2013 vs FER+ top-vote agreement: {np.mean([a == b for a, b in zip(fer_labels, top_names)]):.3f}"
          " (row order is right if this is ~0.6-0.75, it would be ~0.2 if shuffled)")
    if check_against_local(images, fer_labels, usage) < 0.95:
        raise SystemExit("Row order check failed: refusing to build a mislabelled dataset")

    cols = list(FERPLUS_COLS) + ["contempt", "unknown", "NF"]
    votes = plus[cols].to_numpy(dtype=np.float32)
    votes[votes <= 1] = 0  # single votes are annotator noise (FER+ paper)
    keep = plus["Image name"].notna().to_numpy() & (votes.sum(1) > 0)
    keep &= votes.argmax(1) < len(FERPLUS_COLS)  # top vote is one of our 7 emotions
    # reorder the 7 emotion columns into LABELS order
    order = [list(FERPLUS_COLS.values()).index(label) for label in LABELS]
    emo = votes[:, :len(FERPLUS_COLS)][:, order]
    split = np.array([{"Training": 0, "PublicTest": 1, "PrivateTest": 2}[u] for u in usage], dtype=np.int64)

    images, emo, split = images[keep], emo[keep], split[keep]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, images=images, votes=emo, split=split)
    hard = emo.argmax(1)
    print(f"kept {keep.sum()}/{len(keep)}  train {np.sum(split == 0)}  val {np.sum(split == 1)}  test {np.sum(split == 2)}")
    for i, label in enumerate(LABELS):
        print(f"  {label:9s} train {np.sum((hard == i) & (split == 0)):5d}  test {np.sum((hard == i) & (split == 2)):4d}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
