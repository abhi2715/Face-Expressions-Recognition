"""Figures that do not depend on the temporal experiments: architecture, dataset, training curves, confusion matrices, latency, recommender."""
import json, os, re, sys, glob
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

ROOT = os.getcwd(); HERE = os.path.join(ROOT, "Research paper", "MoodLens_paper"); RES = f"{HERE}/results"; FIG = f"{HERE}/figures"
for f in glob.glob("/System/Library/Fonts/Supplemental/Times New Roman*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.6})
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]; CAP = [l.capitalize() for l in LABELS]
C = {"navy": "#1f3b73", "orange": "#d9822b", "green": "#2e8b57", "red": "#b23a48", "grey": "#7a7a7a", "teal": "#2a9d8f", "violet": "#6a4c93"}
M = json.load(open(f"{RES}/model_metrics.json"))
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", pad_inches=0.03); plt.close(fig); print("saved", name)

# ---------------- Fig: architecture ----------------
def box(ax, x, y, w, h, text, fc="#e8eef8", ec=C["navy"], fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.15)
def arrow(ax, a, b, style="-|>", ls="-", c="#333"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, lw=0.8, color=c, linestyle=ls))
fig, ax = plt.subplots(figsize=(7.1, 3.35)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.005, 0.965, "Browser (client)", fontsize=7.5, fontweight="bold", color=C["navy"])
ax.add_patch(FancyBboxPatch((0.0, 0.56), 0.99, 0.41, boxstyle="round,pad=0.005,rounding_size=0.02", fc="#fbfbfb", ec="#999", lw=0.6, ls="--"))
box(ax, 0.02, 0.70, 0.13, 0.18, "Webcam /\nshared call\nwindow", fc="#fff3e0", ec=C["orange"])
box(ax, 0.20, 0.70, 0.14, 0.18, "Frame grab\n640 px (cam)\n1280 px (call)\nJPEG q=0.82", fc="#fff3e0", ec=C["orange"])
box(ax, 0.58, 0.64, 0.17, 0.27, "Temporal layer\n(common.js)\n2 s window average\n1.2 s switch delay\nvalence score", fc="#e9f5ec", ec=C["green"])
box(ax, 0.79, 0.76, 0.195, 0.14, "Mood Menu UI\nscan, combo, cart, QR", fc="#e9f5ec", ec=C["green"])
box(ax, 0.79, 0.585, 0.195, 0.14, "Coach UI\nalerts, flashes, debrief", fc="#e9f5ec", ec=C["green"])
ax.text(0.005, 0.50, "Local server (FastAPI, one per app)", fontsize=7.5, fontweight="bold", color=C["navy"])
ax.add_patch(FancyBboxPatch((0.0, 0.02), 0.99, 0.47, boxstyle="round,pad=0.005,rounding_size=0.02", fc="#fbfbfb", ec="#999", lw=0.6, ls="--"))
box(ax, 0.02, 0.22, 0.12, 0.20, "WebSocket\n/ws", fc="#e8eef8")
box(ax, 0.18, 0.22, 0.14, 0.20, "YuNet face\ndetector\n+ 5 landmarks", fc="#e8eef8")
box(ax, 0.36, 0.22, 0.14, 0.20, "Eye-level\nalign, 48x48\ncrop", fc="#e8eef8")
box(ax, 0.54, 0.22, 0.14, 0.20, "ResNet-18\n(+ flip TTA)\n7 emotions", fc="#e8eef8", bold=True)
box(ax, 0.72, 0.22, 0.12, 0.20, "Face tracker\nIoU match\nEMA a=0.35", fc="#e8eef8")
box(ax, 0.02, 0.05, 0.22, 0.12, "Menu: recommender, orders,\nMood Lift store, QR", fc="#e9f5ec", ec=C["green"], fs=6.3)
box(ax, 0.27, 0.05, 0.20, 0.12, "Coach: playbook,\nAI wingman", fc="#e9f5ec", ec=C["green"], fs=6.3)
box(ax, 0.50, 0.05, 0.22, 0.12, "Chatbot endpoint\n(grounded prompt)", fc="#e9f5ec", ec=C["green"], fs=6.3)
box(ax, 0.86, 0.22, 0.12, 0.20, "JSON:\nprobs, raw,\nbox, id", fc="#e8eef8")
box(ax, 0.77, 0.05, 0.21, 0.12, "Groq API (LLM)\nonly text leaves the PC", fc="#fde8ea", ec=C["red"], fs=6.3)
for a, b in [((0.15, 0.79), (0.20, 0.79)), ((0.34, 0.79), (0.40, 0.79))]: arrow(ax, a, b)
ax.text(0.40, 0.80, "binary WebSocket frames", fontsize=6, ha="left", va="bottom", color="#444")
arrow(ax, (0.27, 0.70), (0.08, 0.42)); arrow(ax, (0.14, 0.32), (0.18, 0.32)); arrow(ax, (0.32, 0.32), (0.36, 0.32)); arrow(ax, (0.50, 0.32), (0.54, 0.32)); arrow(ax, (0.68, 0.32), (0.72, 0.32)); arrow(ax, (0.84, 0.32), (0.86, 0.32))
arrow(ax, (0.92, 0.42), (0.68, 0.66)); arrow(ax, (0.75, 0.80), (0.79, 0.82)); arrow(ax, (0.75, 0.72), (0.79, 0.66))
arrow(ax, (0.72, 0.11), (0.77, 0.11), style="<|-|>", ls="--", c=C["red"])
save(fig, "fig_architecture")

# ---------------- Fig: dataset ----------------
tr = {l: len(os.listdir(f"images/train/{l}")) for l in LABELS}; te = {l: len(os.listdir(f"images/test/{l}")) for l in LABELS}
fig = plt.figure(figsize=(7.1, 3.05)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.55], wspace=0.08)
ax = fig.add_subplot(gs[0]); x = np.arange(7); w = 0.4
b1 = ax.bar(x - w / 2, [tr[l] for l in LABELS], w, label=f"Train ({sum(tr.values()):,})", color=C["navy"]); b2 = ax.bar(x + w / 2, [te[l] for l in LABELS], w, label=f"Test ({sum(te.values()):,})", color=C["orange"])
ax.set_xticks(x); ax.set_xticklabels(CAP, rotation=40, ha="right"); ax.set_ylabel("Images"); ax.legend(frameon=False, loc="upper right")
for r in b1: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 80, f"{int(r.get_height()):,}", ha="center", fontsize=5.6, rotation=90)
ax.set_title("(a) Class distribution (FER-2013)", loc="left"); ax.set_ylim(0, 8600)
sub = gs[1].subgridspec(7, 8, wspace=0.03, hspace=0.03); rng = np.random.default_rng(3)
for r, l in enumerate(LABELS):
    files = sorted(glob.glob(f"images/train/{l}/*.jpg")); pick = rng.choice(len(files), 8, replace=False)
    for c, k in enumerate(pick):
        a = fig.add_subplot(sub[r, c]); a.imshow(cv2.imread(files[k], 0), cmap="gray", vmin=0, vmax=255); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        if c == 0: a.set_ylabel(CAP[r], rotation=0, ha="right", va="center", fontsize=7, labelpad=3)
fig.text(0.40, 0.935, "(b) Random training samples, 48x48 grayscale", fontsize=8.5, ha="left")
save(fig, "fig_dataset")

# ---------------- Fig: FER+ labels ----------------
fp = np.load("cache/ferplus.npz"); te_idx = np.where(fp["split"] == 2)[0]; rng = np.random.default_rng(5)
votes = fp["votes"]; ent = []
fig, axs = plt.subplots(2, 6, figsize=(7.1, 2.6), gridspec_kw={"height_ratios": [1, 0.8], "hspace": 0.1, "wspace": 0.12})
picks = [rng.choice(te_idx[(votes[te_idx].argmax(1) == c)]) for c in (3, 5, 0, 6, 2, 4)]
for j, i in enumerate(picks):
    axs[0, j].imshow(fp["images"][i], cmap="gray"); axs[0, j].axis("off"); v = votes[i] / votes[i].sum()
    axs[1, j].barh(range(7), v[::-1], color=[C["navy"] if k == v.argmax() else "#b8c4de" for k in range(7)][::-1]); axs[1, j].set_xlim(0, 1)
    axs[1, j].set_yticks(range(7)); axs[1, j].set_yticklabels([l[:3] for l in LABELS][::-1] if j == 0 else [], fontsize=5.5); axs[1, j].tick_params(axis="x", labelsize=5.5, pad=1)
    axs[1, j].set_xticks([0, 0.5, 1]); axs[1, j].spines["left"].set_visible(False)
fig.suptitle("Ten annotators vote on each face: FER+ keeps the vote shares as soft training targets", fontsize=8, y=0.99)
save(fig, "fig_ferplus")

# ---------------- Fig: training curves ----------------
def parse(path):
    out = []
    for line in open(path):
        m = re.match(r"epoch\s+(\d+)/(\d+)\s+loss ([\d.]+)(?:\s+train_acc ([\d.]+))?\s+val_acc ([\d.]+)", line)
        if m: out.append((int(m[1]), float(m[3]), float(m[4]) if m[4] else None, float(m[5])))
    return np.array([[x if x is not None else np.nan for x in r] for r in out])
r18, r34, fpl = parse("train_torch.log"), parse("train_resnet34.log"), parse("train_ferplus_resnet18.log")
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.1), gridspec_kw={"wspace": 0.38})
axs[0].plot(r18[:, 0], r18[:, 3] * 100, color=C["navy"], label="ResNet-18"); axs[0].plot(r34[:, 0], r34[:, 3] * 100, color=C["orange"], label="ResNet-34 (aligned mix)")
axs[0].set_title("(a) FER-2013 validation accuracy", loc="left"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Accuracy (%)"); axs[0].legend(frameon=False, loc="lower right")
axs[1].plot(r18[:, 0], r18[:, 1], color=C["navy"], label="train loss"); axs[1].plot(r18[:, 0], 1 - r18[:, 2], color=C["grey"], ls="--", label="1 - train acc")
axs[1].set_title("(b) FER-2013 ResNet-18 training", loc="left"); axs[1].set_xlabel("Epoch"); axs[1].legend(frameon=False)
axs[2].plot(fpl[:, 0], fpl[:, 3] * 100, color=C["green"], label="val acc"); a2 = axs[2].twinx(); a2.plot(fpl[:, 0], fpl[:, 1], color=C["grey"], ls="--", label="soft-CE loss"); a2.spines["right"].set_visible(True); a2.set_ylabel("Loss")
axs[2].set_title("(c) FER+ fine-tuning", loc="left"); axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("Accuracy (%)")
h1, l1 = axs[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); axs[2].legend(h1 + h2, l1 + l2, frameon=False, loc="lower right", fontsize=6)
save(fig, "fig_training")
res = {"r18_epochs": len(r18), "r18_best_val": float(r18[:, 3].max()), "r34_epochs": len(r34), "r34_best_val": float(r34[:, 3].max()), "fp_epochs": len(fpl), "fp_best_val": float(fpl[:, 3].max()),
       "r18_first_val": float(r18[0, 3]), "fp_first_val": float(fpl[0, 3])}
json.dump(res, open(f"{RES}/training_summary.json", "w"), indent=1); print(res)

# ---------------- Fig: confusion matrices ----------------
def cm_plot(ax, cm, title):
    cm = np.array(cm, float); n = cm / cm.sum(1, keepdims=True); im = ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels([l[:3] for l in LABELS], rotation=45, ha="right"); ax.set_yticklabels(CAP)
    for i in range(7):
        for j in range(7): ax.text(j, i, f"{n[i, j] * 100:.0f}", ha="center", va="center", fontsize=6, color="white" if n[i, j] > 0.5 else "#222")
    ax.set_title(title, loc="left"); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); [s.set_visible(True) for s in ax.spines.values()]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"wspace": 0.5})
cm_plot(axs[0], M["FER2013 ResNet-18"]["fer2013_crops_tta"]["confusion"], f"(a) FER-2013 test, ResNet-18 ({M['FER2013 ResNet-18']['fer2013_crops_tta']['acc'] * 100:.1f}%)")
cm_plot(axs[1], M["FER+ ResNet-18 (deployed)"]["ferplus_private_tta"]["confusion"], f"(b) FER+ PrivateTest, deployed ({M['FER+ ResNet-18 (deployed)']['ferplus_private_tta']['acc'] * 100:.1f}%)")
save(fig, "fig_confusion")

# ---------------- Fig: per-class F1 ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.1), gridspec_kw={"wspace": 0.18}); x = np.arange(7); w = 0.27
for k, (name, key, col) in enumerate([("Keras CNN", ("Keras CNN (baseline)", "fer2013_crops"), C["grey"]), ("ResNet-18", ("FER2013 ResNet-18", "fer2013_crops_tta"), C["navy"])]):
    axs[0].bar(x + (k - 0.5) * w * 1.3, np.array(M[key[0]][key[1]]["f1"]) * 100, w * 1.3, label=name, color=col)
axs[0].set_xticks(x); axs[0].set_xticklabels(CAP, rotation=30, ha="right"); axs[0].set_ylabel("F1 (%)"); axs[0].set_title("(a) FER-2013 test (7,178 images)", loc="left"); axs[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0)); axs[0].set_ylim(0, 122)
for k, (name, key, col) in enumerate([("FER-2013 model", ("FER2013 ResNet-18", "ferplus_private_tta"), C["grey"]), ("FER+ raw", ("FER+ ResNet-18 (raw)", "ferplus_private_tta"), C["green"]), ("FER+ deployed", ("FER+ ResNet-18 (deployed)", "ferplus_private_tta"), C["navy"])]):
    axs[1].bar(x + (k - 1) * w, np.array(M[key[0]][key[1]]["f1"]) * 100, w, label=name, color=col)
axs[1].set_xticks(x); axs[1].set_xticklabels(CAP, rotation=30, ha="right"); axs[1].set_title("(b) FER+ PrivateTest (3,519 images)", loc="left"); axs[1].legend(frameon=False, ncol=3, loc="upper center", fontsize=6.3); axs[1].set_ylim(0, 122)
save(fig, "fig_f1")

# ---------------- Fig: latency ----------------
L = json.load(open(f"{RES}/latency.json"))["inprocess"]
fig, ax = plt.subplots(figsize=(3.45, 2.2)); labs = [f"{r['source'].split()[0][:3]}\n{r['faces_in_frame']} face{'s' if r['faces_in_frame'] > 1 else ''}" for r in L]
dec = np.array([r["decode_ms"] for r in L]); det = np.array([r["detect_ms"] for r in L]); cls = np.array([r["classify_ms"] for r in L]); tot = np.array([r["total_ms"] for r in L]); oth = np.maximum(0, tot - dec - det - cls)
ax.bar(range(6), dec, color="#b8c4de", label="JPEG decode"); ax.bar(range(6), det, bottom=dec, color=C["navy"], label="YuNet detect"); ax.bar(range(6), cls, bottom=dec + det, color=C["orange"], label="Align + ResNet")
ax.bar(range(6), oth, bottom=dec + det + cls, color=C["grey"], label="Tracker")
for i, t in enumerate(tot): ax.text(i, t + 1.2, f"{t:.0f}", ha="center", fontsize=6.5)
ax.axhline(71.4, color=C["red"], lw=0.7, ls="--"); ax.text(-0.45, 73.5, "14 fps budget (71 ms)", color=C["red"], fontsize=6, ha="left")
ax.set_xticks(range(6)); ax.set_xticklabels(["cam\n1", "cam\n2", "cam\n4", "call\n1", "call\n2", "call\n4"]); ax.set_ylabel("ms per frame"); ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6, columnspacing=0.8, handlelength=1.2); ax.set_ylim(0, 105)
save(fig, "fig_latency")

# ---------------- Fig: recommender ----------------
R = json.load(open(f"{RES}/recommender.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.1), gridspec_kw={"wspace": 0.25}); moods = list(R["mood_fit"]); x = np.arange(7)
axs[0].bar(x - 0.2, [R["mood_fit"][m]["seek_hit_recommender"] * 100 for m in moods], 0.4, color=C["navy"], label="Recommender"); axs[0].bar(x + 0.2, [R["mood_fit"][m]["seek_hit_random"] * 100 for m in moods], 0.4, color="#b8c4de", label="Random dish per course")
axs[0].set_xticks(x); axs[0].set_xticklabels([m.capitalize() for m in moods], rotation=30, ha="right"); axs[0].set_ylabel("Dishes with a sought tag (%)"); axs[0].set_title("(a) Mood fit of the combo", loc="left"); axs[0].legend(frameon=False, loc="upper center", ncol=2, fontsize=6.3); axs[0].set_ylim(0, 125)
axs[1].bar(x, [R["variety"]["distinct_combos_per_100_seeds"][m] for m in moods], color=C["teal"]); axs[1].set_xticks(x); axs[1].set_xticklabels([m.capitalize() for m in moods], rotation=30, ha="right")
axs[1].set_ylabel("Distinct combos / 100 seeds"); axs[1].set_title("(b) Variety behind Shuffle", loc="left")
for i, m in enumerate(moods): axs[1].text(i, R["variety"]["distinct_combos_per_100_seeds"][m] + 0.3, str(R["variety"]["distinct_combos_per_100_seeds"][m]), ha="center", fontsize=6.5)
save(fig, "fig_recommender")
