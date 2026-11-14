"""Latency of the live pipeline: per-stage timing in-process, then end-to-end over the real WebSocket.
Run with the Mood Menu venv from the SmashGuys-MoodMenu folder:  venv/bin/python <this file> <face_images_dir>
"""
import asyncio, glob, json, os, statistics as st, sys, time
import cv2, numpy as np, torch, websockets
sys.path.insert(0, os.getcwd())
from menu_app.engine import EmotionEngine, align_crop

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "latency.json")
faces_dir = sys.argv[1]
imgs = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(faces_dir, "[a-z]*.jpg"))) if os.path.basename(f) not in ("two.jpg", "face.jpg")]
# a clean face crop from each canvas (290px square at the same place used by the earlier tests)
crops = [im[95:385, 170:470] for im in imgs if im.shape[:2] == (480, 640)]
print(len(crops), "face crops")

def frame(k, W, H, size):
    c = np.full((H, W, 3), 128, np.uint8)
    cols = int(np.ceil(np.sqrt(k))); rows = int(np.ceil(k / cols))
    cw, ch = W // cols, H // rows
    for i in range(k):
        f = cv2.resize(crops[i % len(crops)], (size, size)); r, q = divmod(i, cols)
        x0 = q * cw + (cw - size) // 2; y0 = r * ch + (ch - size) // 2
        c[y0:y0 + size, x0:x0 + size] = f
    return c

eng = EmotionEngine(); tracker = eng.new_tracker()
def sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()
res = {"inprocess": []}
for (W, H, label) in [(640, 480, "camera 640x480"), (1280, 720, "screen 1280x720")]:
    for k in (1, 2, 4):
        size = min(W // 3, 240) if k == 1 else (170 if k == 2 else 130)
        fr = frame(k, W, H, size)
        ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82]); jb = jpg.tobytes()
        t_dec, t_det, t_cls, t_all, n_found = [], [], [], [], []
        for it in range(230):
            t0 = time.perf_counter(); bgr = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR); t1 = time.perf_counter()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            fs = [f for f in eng.detector.detect(bgr) if f[2] >= 40 and f[3] >= 40]; t2 = time.perf_counter()
            raw = eng.classifier.predict([align_crop(gray, f) for f in fs]); sync(); t3 = time.perf_counter()
            tr = eng.new_tracker(); tr.update([tuple(float(v) for v in f[:4]) for f in fs], raw); t4 = time.perf_counter()
            if it >= 30:
                t_dec.append((t1 - t0) * 1e3); t_det.append((t2 - t1) * 1e3); t_cls.append((t3 - t2) * 1e3); t_all.append((t4 - t0) * 1e3); n_found.append(len(fs))
        row = {"source": label, "faces_in_frame": k, "faces_detected_mean": round(st.mean(n_found), 2), "jpeg_kB": round(len(jb) / 1024, 1),
               "decode_ms": round(st.mean(t_dec), 2), "detect_ms": round(st.mean(t_det), 2), "classify_ms": round(st.mean(t_cls), 2),
               "total_ms": round(st.mean(t_all), 2), "p95_ms": round(float(np.percentile(t_all, 95)), 2)}
        res["inprocess"].append(row); print(row, flush=True)

async def ws_bench(port, W, H, n=150):
    fr = frame(1, W, H, 240); ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82]); jb = jpg.tobytes()
    rt, srv = [], []
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None) as ws:
        for i in range(n + 20):
            t0 = time.perf_counter(); await ws.send(jb); r = json.loads(await ws.recv()); dt = (time.perf_counter() - t0) * 1e3
            if i >= 20: rt.append(dt); srv.append(r["ms"])
    return {"port": port, "frames": n, "roundtrip_ms_mean": round(st.mean(rt), 2), "roundtrip_p95_ms": round(float(np.percentile(rt, 95)), 2),
            "server_ms_mean": round(st.mean(srv), 2), "max_fps_serial": round(1000 / st.mean(rt), 1)}
res["websocket"] = {}
for name, port in [("Mood Menu", 8001), ("Conversation Coach", 8000)]:
    try: res["websocket"][name] = asyncio.run(ws_bench(port, 640, 480)); print(name, res["websocket"][name], flush=True)
    except Exception as e: print(name, "ws failed", repr(e))
res["device"] = "Apple GPU (MPS)" if torch.backends.mps.is_available() else "CPU"
json.dump(res, open(OUT, "w"), indent=1)
