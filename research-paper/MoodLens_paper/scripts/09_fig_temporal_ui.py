import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
for f in glob.glob("/System/Library/Fonts/Supplemental/Times New Roman*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 250, "axes.linewidth": 0.6})
HERE = os.getcwd() + "/Research paper/MoodLens_paper"; RES = HERE + "/results"; FIG = HERE + "/figures"; RAW = FIG + "/raw"
C = {"navy": "#1f3b73", "orange": "#d9822b", "green": "#2e8b57", "red": "#b23a48", "grey": "#7a7a7a", "teal": "#2a9d8f"}
T = json.load(open(RES + "/temporal.json"))
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.15), gridspec_kw={"wspace": 0.42, "width_ratios": [0.8, 1, 1.25]})
f = T["steady"]["flips_per_10s"]; v = [f["raw"], f["ema"], f["stable"]]; bars = axs[0].bar(["Raw\nframes", "EMA\n(a=0.35)", "Stable\nmood"], v, color=["#b8c4de", C["navy"], C["orange"]])
for b, x in zip(bars, v): axs[0].text(b.get_x() + b.get_width() / 2, x + 0.3, f"{x:.2f}" if x < 1 else f"{x:.1f}", ha="center", fontsize=7)
axs[0].set_ylabel("Label changes per 10 s"); axs[0].set_title("(a) Flicker on steady faces", loc="left"); axs[0].set_ylim(0, 17)
L = T["transition"]["lags"]
for k, (lab, col) in {"ema": ("EMA a=0.35", C["navy"]), "stable": ("Stable mood", C["orange"])}.items():
    x = np.sort([y for y in L[k] if y is not None]); axs[1].step(x, np.arange(1, len(x) + 1) / len(x) * 100, where="post", color=col, label=lab, lw=1.4)
axs[1].set_xlim(0, 3.2); axs[1].set_xlabel("Seconds until new expression shows"); axs[1].set_ylabel("Transitions (%)"); axs[1].set_title("(b) Switching delay", loc="left"); axs[1].legend(frameon=False, loc="center", bbox_to_anchor=(0.42, 0.55))
axs[1].axvline(T["transition"]["stable"]["median_s"], color=C["orange"], ls=":", lw=0.8); axs[1].text(T["transition"]["stable"]["median_s"] + 0.06, 8, f"median {T['transition']['stable']['median_s']:.1f} s", ha="left", fontsize=6.5, color=C["orange"])
d = T["flash"]; x = d["duration_ms"]
axs[2].axvspan(0, 0.7 * 1000 + 0.5, color="#e8eef8", lw=0); axs[2].plot(x, np.array(d["happy"]) * 100, "o-", ms=3, color=C["navy"], label="steady mood: happy"); axs[2].plot(x, np.array(d["neutral"]) * 100, "s--", ms=3, color=C["orange"], label="steady mood: neutral")
axs[2].axvline(700, color=C["red"], lw=0.8, ls="--"); axs[2].text(715, 28, "0.7 s limit", color=C["red"], fontsize=6.5, rotation=90, va="center")
axs[2].set_xlabel("Flash duration (ms)"); axs[2].set_ylabel("Flashes caught (%)"); axs[2].set_ylim(-3, 108); axs[2].set_title("(c) Micro-expression detection", loc="left"); axs[2].legend(frameon=False, loc="upper right", fontsize=6.3, bbox_to_anchor=(1.02, 0.78))
fig.savefig(FIG + "/fig_temporal.png", bbox_inches="tight", pad_inches=0.03); plt.close(fig); print("temporal ok")

def collage(names, titles, out, cols=2, W=1000, crops=None, figw=7.1, gap=0.03):
    ims = []
    for n, c in zip(names, crops or [None] * len(names)):
        im = Image.open(f"{RAW}/{n}.png").convert("RGB")
        if c: im = im.crop(c)
        ims.append(im.resize((W, int(im.height * W / im.width)), Image.LANCZOS))
    rows = (len(ims) + cols - 1) // cols; hs = [max(i.height for i in ims[r * cols:(r + 1) * cols]) for r in range(rows)]
    fig = plt.figure(figsize=(figw, figw / cols * (sum(hs) / rows) / W * rows + 0.16 * rows)); gs = fig.add_gridspec(rows, cols, height_ratios=hs, wspace=gap, hspace=0.12)
    for i, (im, t) in enumerate(zip(ims, titles)):
        a = fig.add_subplot(gs[i // cols, i % cols]); a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(t, loc="left", fontsize=7.5, pad=2)
        for s in a.spines.values(): s.set_linewidth(0.5); s.set_color("#888")
    fig.savefig(f"{FIG}/{out}.png", bbox_inches="tight", pad_inches=0.02, dpi=230); plt.close(fig); print(out)
collage(["menu_1_scan", "menu_2_filters", "menu_5_lift", "menu_7_table"],
        ["(a) Vibe scan and mood combo", "(b) Filters and a recomputed combo", "(c) Ticket with QR code and Mood Lift", "(d) Table mode, two faces"], "fig_ui_menu")
collage(["coach_1_live", "coach_3_room", "coach_2_flash", "coach_6_debrief"],
        ["(a) Live read, playbook and \"Their vibe\"", "(b) Room mode with two faces", "(c) Micro-expression and alert feeds", "(d) Post-conversation debrief"], "fig_ui_coach",
        crops=[None, None, None, (0, 0, 2160, 1500)])
collage(["menu_8_chat", "coach_5_chat"], ["(a) Smashy, with Add buttons", "(b) Coach, with live mood context"], "fig_ui_chat", crops=[(1380, 380, 2160, 1410), (1450, 360, 2160, 1500)], W=700, figw=3.45, gap=0.04)
