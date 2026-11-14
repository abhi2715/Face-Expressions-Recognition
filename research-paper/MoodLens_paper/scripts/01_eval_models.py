"""Score every shipped checkpoint (and the original Keras CNN) on FER-2013 and FER+ test data.
Run from the Face-Expressions-Recognition-main folder:  venv/bin/python "Research paper/MoodLens_paper/scripts/01_eval_models.py"
"""
import json, os, sys, time
import numpy as np, torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from train_torch import EmotionNet, FaceDataset, LABELS, TEST_DIR, load_dataset

OUT = "Research paper/MoodLens_paper/results"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

x_test, y_test = load_dataset(TEST_DIR)
aligned = np.load("cache/aligned_test.npz")
fp = np.load("cache/ferplus.npz")
te = fp["split"] == 2
x_fp, y_fp = fp["images"][te], fp["votes"][te].argmax(1)
print("FER2013 test", x_test.shape, "aligned", aligned["images"].shape, "found", int(aligned["found"].sum()), "FER+ test", x_fp.shape)

@torch.no_grad()
def probs_of(model, images, targets, tta):
    loader = DataLoader(FaceDataset(images, targets, False), batch_size=256)
    out = []
    for x, _ in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta: p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()

def metrics(p, y):
    pred = p.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=range(7), zero_division=0)
    return {"acc": float((pred == y).mean()), "macro_f1": float(f1.mean()), "precision": pr.tolist(), "recall": rc.tolist(),
            "f1": f1.tolist(), "support": sup.tolist(), "confusion": confusion_matrix(y, pred, labels=range(7)).tolist()}

models = {"FER2013 ResNet-18": "models/emotion_resnet18.pt", "FER+ ResNet-18 (raw)": "models/emotion_resnet18_ferplus_raw.pt",
          "FER+ ResNet-18 (deployed)": "models/emotion_resnet18_ferplus.pt"}
sets = {"fer2013_crops": (x_test, y_test), "fer2013_aligned": (aligned["images"], y_test), "ferplus_private": (x_fp, y_fp)}
res = {}
for name, path in models.items():
    ck = torch.load(path, map_location="cpu")
    m = EmotionNet(ck.get("arch", "resnet18"), pretrained=False); m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    res[name] = {"params_M": sum(p.numel() for p in m.parameters()) / 1e6, "file_MB": os.path.getsize(path) / 1e6, "val_acc_ckpt": ck.get("val_acc")}
    for sname, (imgs, y) in sets.items():
        for tta in (False, True):
            p = probs_of(m, imgs, y, tta)
            r = metrics(p, y)
            res[name][f"{sname}{'_tta' if tta else ''}"] = r
            print(f"{name:28s} {sname:16s} tta={tta!s:5s} acc={r['acc']:.4f} macroF1={r['macro_f1']:.4f}", flush=True)

# Keras baseline (the project's first model)
try:
    import keras
    km = keras.models.load_model("facialemotionmodel.keras")
    xk = (x_test.astype("float32") / 255.0)[..., None]
    t0 = time.time(); pk = km.predict(xk, batch_size=256, verbose=0); dt = time.time() - t0
    r = metrics(pk, y_test)
    res["Keras CNN (baseline)"] = {"params_M": km.count_params() / 1e6, "file_MB": os.path.getsize("facialemotionmodel.keras") / 1e6, "fer2013_crops": r}
    print(f"Keras baseline acc={r['acc']:.4f} macroF1={r['macro_f1']:.4f} ({dt:.1f}s)")
except Exception as e:
    print("keras baseline skipped:", repr(e))
res["_meta"] = {"labels": LABELS, "n_fer2013_test": int(len(y_test)), "n_ferplus_test": int(len(y_fp)), "aligned_found": int(aligned["found"].sum())}
json.dump(res, open(f"{OUT}/model_metrics.json", "w"), indent=1)
print("saved")
