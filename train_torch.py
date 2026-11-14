"""Train a high-accuracy expression model (ImageNet-pretrained ResNet) on the Apple GPU.

Usage:  venv/bin/python train_torch.py [--arch resnet18|resnet34] [--epochs 40] [--aligned-mix 0.5] [--out PATH]
Output: models/emotion_<arch>.pt (best checkpoint on a held-out validation split)

--aligned-mix trains on a mix of the dataset crops and the app-style aligned crops from
prepare_aligned.py, so the model also learns the framing it sees from the live camera.
images/test is never used for model selection; it is only scored once at the end.
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
ALIGNED_TRAIN = "cache/aligned_train.npz"
ALIGNED_TEST = "cache/aligned_test.npz"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 48
NET_SIZE = 112  # faces are upsampled so the pretrained filters see familiar scales
ARCHS = {"resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1), "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1)}


def load_dataset(directory):
    images, targets = [], []
    for idx, label in enumerate(LABELS):
        label_dir = os.path.join(directory, label)
        for name in sorted(os.listdir(label_dir)):
            if name.startswith(".") or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(label_dir, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            targets.append(idx)
    return np.stack(images), np.array(targets, dtype=np.int64)


class FaceDataset(Dataset):
    """Serves the dataset crop, or (with probability aligned_mix) the app-style aligned crop."""

    def __init__(self, images, targets, train, aligned=None, aligned_mix=0.0):
        self.images = torch.from_numpy(images).unsqueeze(1)  # N,1,48,48 uint8
        self.aligned = torch.from_numpy(aligned).unsqueeze(1) if aligned is not None else None
        self.aligned_mix = aligned_mix
        self.targets = torch.from_numpy(targets)
        self.train = train
        self.tf = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=12, translate=(0.06, 0.06)),
            transforms.ColorJitter(brightness=0.35, contrast=0.35),
        ]) if train else None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        use_aligned = self.aligned is not None and torch.rand(1).item() < self.aligned_mix
        x = self.aligned[i] if use_aligned else self.images[i]
        if self.tf is not None:
            x = self.tf(x)
        x = x.float() / 255.0
        if self.train and torch.rand(1) < 0.4:
            x = transforms.functional.erase(x, *self._erase_box(), v=0.0)
        return x, self.targets[i]

    @staticmethod
    def _erase_box():
        h = int(torch.randint(8, 20, (1,)))
        w = int(torch.randint(8, 20, (1,)))
        top = int(torch.randint(0, IMG_SIZE - h, (1,)))
        left = int(torch.randint(0, IMG_SIZE - w, (1,)))
        return top, left, h, w


class EmotionNet(nn.Module):
    """ImageNet ResNet that takes 1x48x48 grayscale faces in [0, 1]."""

    def __init__(self, arch="resnet18", pretrained=True):
        super().__init__()
        build, weights = ARCHS[arch]
        self.backbone = build(weights=weights if pretrained else None)
        self.backbone.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(self.backbone.fc.in_features, len(LABELS)))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(NET_SIZE, NET_SIZE), mode="bilinear", align_corners=False)
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        return self.backbone(x)


@torch.no_grad()
def predict(model, loader, device, tta=True):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta:
            p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
        probs.append(p.cpu())
        targets.append(y)
    return torch.cat(probs), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCHS), default="resnet18")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--aligned-mix", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = args.out or f"models/emotion_{args.arch}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}  arch: {args.arch}  aligned_mix: {args.aligned_mix}", flush=True)

    x_all, y_all = load_dataset(TRAIN_DIR)
    x_test, y_test = load_dataset(TEST_DIR)
    a_all = a_test = None
    if args.aligned_mix > 0:
        a_all = np.load(ALIGNED_TRAIN)["images"]
        cached = np.load(ALIGNED_TEST)
        a_test, found_test = cached["images"], cached["found"]
        assert len(a_all) == len(x_all) and len(a_test) == len(x_test), "aligned cache is stale: rerun prepare_aligned.py"

    # Stratified hold-out split from train for checkpoint selection (fixed seed so models share it)
    rng = np.random.default_rng(42)
    val_idx = np.concatenate([
        rng.permutation(np.where(y_all == c)[0])[: int(np.sum(y_all == c) * args.val_frac)]
        for c in range(len(LABELS))
    ])
    train_mask = np.ones(len(y_all), dtype=bool)
    train_mask[val_idx] = False
    x_train, y_train, x_val, y_val = x_all[train_mask], y_all[train_mask], x_all[val_idx], y_all[val_idx]
    a_train = a_all[train_mask] if a_all is not None else None
    print(f"train {len(y_train)}  val {len(y_val)}  test {len(y_test)}", flush=True)

    train_loader = DataLoader(FaceDataset(x_train, y_train, True, a_train, args.aligned_mix), batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loaders = [DataLoader(FaceDataset(x_val, y_val, False), batch_size=256)]
    if a_all is not None:  # select on both framings so the checkpoint is good for dataset and camera crops
        val_loaders.append(DataLoader(FaceDataset(a_all[val_idx], y_val, False), batch_size=256))

    # Softened class weights so rare classes (disgust) aren't ignored
    counts = np.bincount(y_train, minlength=len(LABELS))
    class_weight = torch.tensor(np.sqrt(counts.sum() / (len(LABELS) * counts)), dtype=torch.float32, device=device)

    model = EmotionNet(args.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.15)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Mixup on half the batches: regularises the noisy FER labels
            if np.random.rand() < 0.5:
                lam = float(np.random.beta(0.4, 0.4))
                perm = torch.randperm(x.size(0), device=device)
                logits = model(lam * x + (1 - lam) * x[perm])
                loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[perm])
            else:
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        accs = []
        for loader in val_loaders:
            val_probs, val_targets = predict(model, loader, device)
            accs.append((val_probs.argmax(1) == val_targets).float().mean().item())
        val_acc = float(np.mean(accs))
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "labels": LABELS, "arch": args.arch, "val_acc": val_acc},
                       out_path)
            marker = "  * saved"
        detail = " / ".join(f"{a:.4f}" for a in accs)
        print(f"epoch {epoch:2d}/{args.epochs}  loss {loss_sum / total:.4f}  train_acc {correct / total:.4f}  "
              f"val_acc {val_acc:.4f} ({detail})  ({time.time() - start:.0f}s){marker}", flush=True)

    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_sets = [("dataset crops", x_test)]
    if a_test is not None:
        test_sets.append(("aligned crops", a_test))
    for name, images in test_sets:
        loader = DataLoader(FaceDataset(images, y_test, False), batch_size=256)
        probs, targets = predict(model, loader, device)
        preds = probs.argmax(1)
        print(f"\nTest accuracy on {name} (flip TTA): {(preds == targets).float().mean():.4f}")
        for i, label in enumerate(LABELS):
            mask = targets == i
            print(f"  {label:9s} {(preds[mask] == i).float().mean():.3f}  (n={int(mask.sum())})")
    print(f"Saved best model (val_acc {best_val:.4f}) to {out_path}")


if __name__ == "__main__":
    main()
