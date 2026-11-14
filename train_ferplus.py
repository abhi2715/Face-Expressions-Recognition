"""Fine-tune the expression model on FER+ (cleaner 10-annotator labels for the FER-2013 faces).

Usage:  venv/bin/python train_ferplus.py [--arch resnet18] [--init models/emotion_resnet18.pt] [--epochs 30]
Needs:  cache/ferplus.npz from prepare_ferplus.py
Output: models/emotion_<arch>_ferplus.pt (same checkpoint format as train_torch.py, so the apps load it as is)

Trains on the annotators' vote distribution (soft labels), picks the checkpoint on the FER+ PublicTest split,
and scores the PrivateTest split once at the end against the majority label (the standard FER+ metric).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_torch import ARCHS, IMG_SIZE, LABELS, EmotionNet

CACHE = "cache/ferplus.npz"


class SoftFaceDataset(Dataset):
    def __init__(self, images, targets, train):
        self.images = torch.from_numpy(images).unsqueeze(1)  # N,1,48,48 uint8
        self.targets = torch.from_numpy(targets)  # N,7 probabilities
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
        x = self.images[i]
        if self.tf is not None:
            x = self.tf(x)
        x = x.float() / 255.0
        if self.train and torch.rand(1) < 0.4:
            h, w = int(torch.randint(8, 20, (1,))), int(torch.randint(8, 20, (1,)))
            top, left = int(torch.randint(0, IMG_SIZE - h, (1,))), int(torch.randint(0, IMG_SIZE - w, (1,)))
            x = transforms.functional.erase(x, top, left, h, w, v=0.0)
        return x, self.targets[i]


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, 1)).sum(1).mean()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device)
        p = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
        probs.append(p.cpu())
        targets.append(y)
    return torch.cat(probs), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCHS), default="resnet18")
    parser.add_argument("--init", default=None, help="start from this checkpoint (e.g. the FER-2013 model)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = args.out or f"models/emotion_{args.arch}_ferplus.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data = np.load(CACHE)
    images, votes, split = data["images"], data["votes"], data["split"]
    targets = votes / votes.sum(1, keepdims=True)
    tr, va, te = (split == 0), (split == 1), (split == 2)
    print(f"device: {device}  arch: {args.arch}  init: {args.init}  "
          f"train {tr.sum()}  val {va.sum()}  test {te.sum()}", flush=True)

    train_loader = DataLoader(SoftFaceDataset(images[tr], targets[tr], True), batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(SoftFaceDataset(images[va], targets[va], False), batch_size=256)
    test_loader = DataLoader(SoftFaceDataset(images[te], targets[te], False), batch_size=256)

    model = EmotionNet(args.arch, pretrained=args.init is None)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu")
        assert ckpt.get("arch", "resnet18") == args.arch and ckpt.get("labels", LABELS) == LABELS
        model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.15)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, loss_sum = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if np.random.rand() < 0.5:  # mixup on half the batches
                lam = float(np.random.beta(0.4, 0.4))
                perm = torch.randperm(x.size(0), device=device)
                x, y = lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
            loss = soft_ce(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

        probs, tgt = predict(model, val_loader, device)
        val_acc = (probs.argmax(1) == tgt.argmax(1)).float().mean().item()
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "labels": LABELS, "arch": args.arch, "val_acc": val_acc,
                        "dataset": "FER+"}, out_path)
            marker = "  * saved"
        print(f"epoch {epoch:2d}/{args.epochs}  loss {loss_sum / total:.4f}  val_acc {val_acc:.4f}  "
              f"({time.time() - start:.0f}s){marker}", flush=True)

    model.load_state_dict(torch.load(out_path, map_location=device)["state_dict"])
    probs, tgt = predict(model, test_loader, device)
    preds, truth = probs.argmax(1), tgt.argmax(1)
    print(f"\nFER+ test accuracy (PrivateTest, majority label, flip TTA): {(preds == truth).float().mean():.4f}")
    for i, label in enumerate(LABELS):
        mask = truth == i
        print(f"  {label:9s} {(preds[mask] == i).float().mean():.3f}  (n={int(mask.sum())})")
    print(f"Saved best model (val_acc {best_val:.4f}) to {out_path}")


if __name__ == "__main__":
    main()
