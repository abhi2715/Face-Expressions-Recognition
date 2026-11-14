"""Train the facial expression model on images/train and evaluate on images/test.

Usage:  venv/bin/python train.py [--epochs 60] [--batch-size 128]
Output: facialemotionmodel.keras (best checkpoint by validation accuracy)
"""
import argparse
import os

import cv2
import numpy as np
import keras
from keras import layers

TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
MODEL_PATH = "facialemotionmodel.keras"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 48


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
        print(f"{directory}/{label}: loaded")
    x = np.array(images, dtype=np.float32)[..., np.newaxis] / 255.0
    y = np.array(targets, dtype=np.int32)
    return x, y


def conv_block(x, filters, dropout):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    # Augmentation layers are only active during training
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    x = conv_block(x, 32, 0.2)
    x = conv_block(x, 64, 0.25)
    x = conv_block(x, 128, 0.3)
    x = conv_block(x, 256, 0.35)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)
    return keras.Model(inputs, outputs, name="emotion_cnn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    x_train, y_train = load_dataset(TRAIN_DIR)
    x_test, y_test = load_dataset(TEST_DIR)
    print(f"train: {x_train.shape}, test: {x_test.shape}")

    # Softened class weights so rare classes (disgust) aren't ignored
    counts = np.bincount(y_train, minlength=len(LABELS))
    weights = np.sqrt(counts.sum() / (len(LABELS) * counts))
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("class weights:", {LABELS[i]: round(w, 2) for i, w in class_weight.items()})

    y_train_oh = keras.utils.to_categorical(y_train, len(LABELS))
    y_test_oh = keras.utils.to_categorical(y_test, len(LABELS))

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
    ]
    model.fit(
        x_train, y_train_oh,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test_oh),
        class_weight=class_weight,
        callbacks=callbacks,
    )

    best = keras.models.load_model(MODEL_PATH)
    _, acc = best.evaluate(x_test, y_test_oh, verbose=0)
    print(f"\nBest model test accuracy: {acc:.4f}  (saved to {MODEL_PATH})")

    preds = best.predict(x_test, verbose=0).argmax(axis=1)
    for i, label in enumerate(LABELS):
        mask = y_test == i
        print(f"  {label:9s} {np.mean(preds[mask] == i):.3f}  (n={mask.sum()})")


if __name__ == "__main__":
    keras.utils.set_random_seed(42)
    main()
