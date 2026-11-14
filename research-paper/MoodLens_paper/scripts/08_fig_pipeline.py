"""Fig: one real two-person frame -> YuNet boxes + eye landmarks -> aligned 48x48 crops -> class probabilities."""
import os, sys, glob
import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, os.getcwd())
from app.engine import FaceDetector, align_crop
from train_torch import EmotionNet, LABELS
for f in glob.glob("/System/Library/Fonts/Supplemental/Times New Roman*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "savefig.dpi": 300, "axes.linewidth": 0.6})
HERE = "Research paper/MoodLens_paper"
frame = cv2.imread(f"{HERE}/results/test_faces/two.jpg"); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
det = FaceDetector(); faces = det.detect(frame); faces = sorted(faces, key=lambda f: f[0])
ck = torch.load("models/emotion_resnet18_ferplus.pt", map_location="cpu"); net = EmotionNet("resnet18", pretrained=False); net.load_state_dict(ck["state_dict"]); net.eval()
crops = [align_crop(gray, f) for f in faces]
with torch.no_grad():
    x = torch.from_numpy(np.stack(crops)).float().div(255).unsqueeze(1); p = ((net(x).softmax(1) + net(torch.flip(x, dims=[3])).softmax(1)) / 2).numpy()
fig = plt.figure(figsize=(7.1, 2.35)); gs = fig.add_gridspec(2, 4, width_ratios=[2.1, 0.85, 0.85, 1.9], wspace=0.28, hspace=0.35)
a = fig.add_subplot(gs[:, 0]); rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); a.imshow(rgb); a.axis("off")
cols = ["#ffd54a", "#22d3ee"]
for k, (x0, y0, w, h, eyes) in enumerate(faces):
    a.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=cols[k], lw=1.6))
    for (ex, ey) in eyes: a.plot(ex, ey, "o", ms=2.6, color="#e63946")
    a.text(x0, y0 - 6, f"#{k + 1}", color=cols[k], fontsize=8, fontweight="bold", va="bottom")
a.set_title("(a) YuNet boxes and eye points", fontsize=8, loc="left")
for k in range(2):
    b = fig.add_subplot(gs[k, 1]); b.imshow(crops[k], cmap="gray"); b.set_xticks([]); b.set_yticks([]); [s.set_edgecolor(cols[k]) for s in b.spines.values()]; [s.set_linewidth(1.5) for s in b.spines.values()]
    if k == 0: b.set_title("(b) 48x48\naligned crop", fontsize=7.5, loc="left")
    c = fig.add_subplot(gs[k, 2]); c.imshow(cv2.resize(crops[k], (48, 48), interpolation=cv2.INTER_NEAREST)[::-1][::-1], cmap="gray", alpha=0)  # spacer for alignment
    c.axis("off")
    d = fig.add_subplot(gs[k, 3]); v = p[k]; d.barh(range(7), v[::-1] * 100, color=[cols[k] if i == v.argmax() else "#c9d1e3" for i in range(7)][::-1], ec="#555", lw=0.4)
    d.set_yticks(range(7)); d.set_yticklabels([l.capitalize() for l in LABELS][::-1], fontsize=6.5); d.set_xlim(0, 100); d.tick_params(axis="x", labelsize=6.5)
    for s in ("top", "right"): d.spines[s].set_visible(False)
    d.text(min(v.max() * 100 + 2, 62), 6 - int(v.argmax()), f"{v.max() * 100:.0f}%", va="center", fontsize=6.5)
    if k == 0: d.set_title("(c) Class probabilities (%)", fontsize=7.5, loc="left")
fig.savefig(f"{HERE}/figures/fig_pipeline.png", bbox_inches="tight", pad_inches=0.03); print("saved", [LABELS[i] for i in p.argmax(1)], len(faces))
