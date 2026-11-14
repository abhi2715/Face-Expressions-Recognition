"""Simulated live-video experiments that reuse the apps' real smoothing / micro-expression rules.
Frames are jittered copies of app-aligned test faces (shift, rotation, scale, brightness, sensor noise), classified by the
deployed model, so flicker comes from the model's real sensitivity. Rules ported from common.js / coach.js / engine.py.
Run from Face-Expressions-Recognition-main:  venv/bin/python "Research paper/MoodLens_paper/scripts/03_temporal.py"
"""
import json, os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.getcwd())
from train_torch import EmotionNet, LABELS

OUT = "Research paper/MoodLens_paper/results/temporal.json"
rng = np.random.default_rng(7)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load("models/emotion_resnet18_ferplus.pt", map_location="cpu")
net = EmotionNet("resnet18", pretrained=False); net.load_state_dict(ck["state_dict"]); net.to(dev).eval()
al = np.load("cache/aligned_test.npz"); imgs, found = al["images"], al["found"]
ytrue = np.array(sum(([i] * n for i, n in enumerate([958, 111, 1024, 1774, 1233, 1247, 831])), []))
FPS = 14.0; DT = 1000.0 / FPS            # ms between processed frames (≈ what the apps reach)
ALPHA, WIN, SWITCH, MIN_FRAME = 0.35, 2000.0, 1200.0, 70.0

@torch.no_grad()
def predict(frames):
    out = []
    for i in range(0, len(frames), 1024):
        x = torch.from_numpy(np.stack(frames[i:i + 1024])).float().div(255).unsqueeze(1).to(dev)
        p = (net(x).softmax(1) + net(torch.flip(x, dims=[3])).softmax(1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)

def jitter(img):
    ang, sc = rng.uniform(-5, 5), rng.uniform(0.95, 1.05)
    m = cv2.getRotationMatrix2D((24, 24), ang, sc); m[:, 2] += rng.uniform(-2, 2, 2)
    f = cv2.warpAffine(img, m, (48, 48), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    f = f * rng.uniform(0.8, 1.2) + rng.normal(0, 5, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)

def seq_probs(sources):
    """sources: list of 48x48 images, one per frame -> raw probs per frame"""
    return predict([jitter(s) for s in sources])

def ema(raw):
    out = np.empty_like(raw); out[0] = raw[0]
    for i in range(1, len(raw)): out[i] = ALPHA * raw[i] + (1 - ALPHA) * out[i - 1]
    return out

def stable_series(sm):
    """common.js updateStableMood on the smoothed history; returns the stable mood per frame (+ since-time)"""
    stable, cand, cand_since, since = None, None, 0.0, 0.0
    res, sinces = [], []
    for i in range(len(sm)):
        now = i * DT
        lo = max(0, int(np.ceil((now - WIN) / DT)))
        cur = sm[lo:i + 1].mean(0); top = int(cur.argmax())
        if stable is None: stable, since = top, now
        elif top != stable:
            if cand != top: cand, cand_since = top, now
            elif now - cand_since > SWITCH: stable, since, cand = top, now, None
        else: cand = None
        res.append(stable); sinces.append(since)
    return np.array(res), np.array(sinces)

def changes(a): return int((np.diff(a) != 0).sum())

# ---------- confident, correctly classified faces (for transitions / flashes) ----------
idx_all = np.where(found)[0]
p_all = predict([imgs[i] for i in idx_all])
good = {c: idx_all[(p_all.argmax(1) == c) & (ytrue[idx_all] == c) & (p_all.max(1) > 0.7)] for c in range(7)}
print({LABELS[c]: len(v) for c, v in good.items()})
res = {"fps_assumed": FPS, "confident_faces": {LABELS[c]: int(len(v)) for c, v in good.items()}}

# ---------- A. steady sequences: label flicker ----------
N_A, FR_A = 280, int(10 * FPS)
rows = []
t0 = time.time()
for s in range(N_A):
    c = s % 7; i = rng.choice(np.where(found & (ytrue == c))[0])
    raw = seq_probs([imgs[i]] * FR_A); sm = ema(raw); st, _ = stable_series(sm)
    rows.append((changes(raw.argmax(1)), changes(sm.argmax(1)), changes(st), (raw.argmax(1) == c).mean(), (sm.argmax(1) == c).mean(), (st == c).mean(),
                 float(raw.max(1).mean())))
rows = np.array(rows)
res["steady"] = {"sequences": N_A, "seconds_each": 10, "flips_per_10s": {"raw": rows[:, 0].mean(), "ema": rows[:, 1].mean(), "stable": rows[:, 2].mean()},
                 "seq_with_any_flip_pct": {"raw": float((rows[:, 0] > 0).mean() * 100), "ema": float((rows[:, 1] > 0).mean() * 100), "stable": float((rows[:, 2] > 0).mean() * 100)},
                 "frame_acc": {"raw": rows[:, 3].mean(), "ema": rows[:, 4].mean(), "stable": rows[:, 5].mean()}}
print("A", res["steady"], f"{time.time()-t0:.0f}s", flush=True)

# ---------- B. transitions A -> B: lag ----------
lags = {"raw": [], "ema": [], "stable": []}; N_B = 210; PRE, POST = int(5 * FPS), int(7 * FPS)
for s in range(N_B):
    a, b = rng.choice(7, 2, replace=False)
    ia, ib = rng.choice(good[a]), rng.choice(good[b])
    raw = seq_probs([imgs[ia]] * PRE + [imgs[ib]] * POST); sm = ema(raw); st, _ = stable_series(sm)
    def lag(series):
        for k in range(PRE, len(series) - 4):
            if (series[k:k + 5] == b).all(): return (k - PRE) * DT / 1000
        return np.nan
    lags["raw"].append(lag(raw.argmax(1))); lags["ema"].append(lag(sm.argmax(1))); lags["stable"].append(lag(st))
res["transition"] = {"pairs": N_B, **{k: {"median_s": float(np.nanmedian(v)), "p90_s": float(np.nanpercentile(v, 90)), "reached_pct": float((~np.isnan(v)).mean() * 100)} for k, v in lags.items()}}
res["transition"]["lags"] = {k: [None if np.isnan(x) else round(float(x), 3) for x in v] for k, v in lags.items()}
print("B", {k: res["transition"][k] for k in lags}, flush=True)

# ---------- C. micro-expression detector ----------
def detect(raw, base_idx, fpsdt=DT):
    """coach.js detectFlash on raw probs; returns list of (emotion_idx, dur_ms)"""
    events, run, stable_since = [], None, 0.0
    sm = ema(raw); st, since = stable_series(sm)
    for i in range(len(raw)):
        now = i * DT; base = st[i]
        if now - since[i] < 1000: run = None; continue
        top = int(raw[i].argmax())
        if top != base and top != 4 and raw[i][top] > 0.45:
            if run is not None and run["e"] == top: run["last"] = now; run["n"] += 1
            else: run = {"e": top, "start": now, "last": now, "n": 1, "base": base}
        elif run is not None:
            dur = run["last"] - run["start"] + MIN_FRAME
            if run["n"] >= 2 and dur <= 700 and top == run["base"]: events.append((run["e"], dur))
            run = None
    return events

FLASH_E = [0, 1, 2, 5, 6]               # angry disgust fear sad surprise
det = {}; TR = 60
for base_c in (3, 4):                    # happy and neutral baselines
    for F in range(1, 16):
        hit = 0; tot = 0
        for _ in range(TR):
            e = int(rng.choice([x for x in FLASH_E if x != base_c])); ib, ie = rng.choice(good[base_c]), rng.choice(good[e])
            src = [imgs[ib]] * 50 + [imgs[ie]] * F + [imgs[ib]] * 30
            raw = seq_probs(src); ev = detect(raw, base_c)
            tot += 1; hit += any(x[0] == e for x in ev)
        det[f"{LABELS[base_c]}|{F}"] = hit / tot
    print("C base", LABELS[base_c], [round(det[f"{LABELS[base_c]}|{F}"], 2) for F in range(1, 16)], flush=True)
res["flash"] = {"frames": list(range(1, 16)), "duration_ms": [round((F - 1) * DT + MIN_FRAME) for F in range(1, 16)], "trials_each": TR,
                **{LABELS[b]: [det[f"{LABELS[b]}|{F}"] for F in range(1, 16)] for b in (3, 4)}}
# false alarms on steady faces, 30 s each
FA_SEQ, FA_LEN = 90, int(30 * FPS); fa = 0
for s in range(FA_SEQ):
    c = [0, 2, 3, 4, 5, 6][s % 6]; i = rng.choice(good[c])
    ev = detect(seq_probs([imgs[i]] * FA_LEN), c); fa += len(ev)
res["flash"]["false_alarms"] = {"sequences": FA_SEQ, "minutes": FA_SEQ * 0.5, "events": int(fa), "per_minute": fa / (FA_SEQ * 0.5)}
print("C false alarms", res["flash"]["false_alarms"], flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=float)
