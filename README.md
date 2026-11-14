# Face Expression Recognition

Real-time facial expression recognition for seven emotions (angry, disgust, fear, happy, neutral, sad, surprise).
It starts from a small Keras CNN on FER-2013 and ends with an ImageNet-pretrained ResNet-18 fine-tuned on FER+, with
a face detector and alignment step so it works on live camera frames.

## Results

| Model | Test set | Accuracy | Macro-F1 |
|---|---|---|---|
| Keras CNN (first model) | FER-2013 test (7,178) | 64.25 % | 0.611 |
| ResNet-18 | FER-2013 test | 69.99 % | 0.690 |
| ResNet-18, FER+ fine-tuned | FER+ PrivateTest (3,519) | 85.08 % | 0.766 |
| ResNet-18, FER+ (shipped checkpoint) | FER+ PrivateTest | 83.38 % | 0.755 |

About 23 ms per frame on an Apple GPU (face detection 12 ms, alignment and classification 10 ms).

## How it works (`app/engine.py`)

1. **Face detection**: YuNet finds each face (at least 40x40 px) and five landmarks. Falls back to OpenCV Haar cascade
   if the YuNet file is missing.
2. **Alignment**: rotate so the eyes are level, then crop to FER-2013 framing (1.25x face width, eyes 40 % down) as a
   48x48 grayscale patch.
3. **Classification**: ResNet-18 gives probabilities for the seven emotions. The crop is also mirrored and the two
   results are averaged (flip test-time augmentation). Several `models/emotion_*.pt` files are averaged as an ensemble.
4. **Tracking and smoothing**: faces keep a stable id (IoU matching) and an exponentially smoothed probability vector.

## Run

```bash
python3.12 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python realtimedetection.py          # webcam window, press q or ESC to quit
```

## Training

| Script | What it does |
|---|---|
| `train.py` | first model: small Keras CNN (4 conv blocks) on FER-2013, writes `facialemotionmodel.keras` |
| `train_torch.py` | pretrained ResNet-18/34 in PyTorch: augmentation (crop, flip, rotation, brightness, erasing, mixup), label smoothing, class weights, AdamW one-cycle; best epoch chosen on a 10 % validation split, test set scored once |
| `prepare_aligned.py` | re-crops dataset faces the way the live detector does so the model can train on camera-style crops |
| `prepare_ferplus.py` | builds the FER+ set (10-annotator relabelling), dropping contempt, unknown and not-a-face |
| `train_ferplus.py` | fine-tunes on FER+ vote shares with soft cross-entropy |
| `evaluate_ensemble.py` | scores models and their average |
| `trainmodel.ipynb` | notebook version of the Keras training |

Datasets are not included. Put FER-2013 in `images/train` and `images/test` (one folder per emotion) and FER+ labels
in `data/ferplus_raw/`. Checkpoints in `models/`: `emotion_resnet18.pt` (FER-2013), `emotion_resnet18_ferplus.pt`
(FER+), and `face_detection_yunet_2023mar.onnx`.

## Limits

- Expression is not emotion: the model reads facial movement, not feelings. Treat output as a hint.
- Fear and sadness are the weakest classes; fear is often confused with surprise.
- FER-2013 faces are small grayscale web photos, so accuracy varies with lighting, angle, age, skin tone and glasses.
