"""Score each trained model and their averaged ensemble on the test set.

Usage:  venv/bin/python evaluate_ensemble.py models/emotion_resnet18.pt models/emotion_resnet34.pt
Reports accuracy on the dataset crops and on the app-style aligned crops (cache/aligned_test.npz).
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_torch import ALIGNED_TEST, LABELS, TEST_DIR, EmotionNet, FaceDataset, load_dataset, predict


def main(paths):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x_test, y_test = load_dataset(TEST_DIR)
    sets = {"dataset crops": x_test, "aligned crops": np.load(ALIGNED_TEST)["images"]}
    results = {name: [] for name in sets}
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        model = EmotionNet(ckpt.get("arch", "resnet18"), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        for name, images in sets.items():
            probs, targets = predict(model, DataLoader(FaceDataset(images, y_test, False), batch_size=256), device)
            results[name].append(probs)
            print(f"{path:34s} {name:14s} {(probs.argmax(1) == targets).float().mean():.4f}")
    targets = torch.from_numpy(y_test)
    for name, probs in results.items():
        ens = torch.stack(probs).mean(0)
        preds = ens.argmax(1)
        print(f"\nENSEMBLE ({len(paths)} models) on {name}: {(preds == targets).float().mean():.4f}")
        for i, label in enumerate(LABELS):
            mask = targets == i
            print(f"  {label:9s} {(preds[mask] == i).float().mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["models/emotion_resnet18.pt", "models/emotion_resnet34.pt"])
