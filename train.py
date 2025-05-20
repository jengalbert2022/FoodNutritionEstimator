#!/usr/bin/env python3
"""
train_food101.py
----------------
End-to-end mixed-precision training script for Food-101 laid out as:

food-101/
    images/<class_name>/<image_id>.jpg
    meta/
        classes.txt
        train.txt
        test.txt
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import SGD

# mixed precision
mixed_precision.set_global_policy("mixed_float16")  # automatic dynamic scaling

# ───────────────────── paths & constants ────────────────────
ROOT_DIR   = "food-101"
IMG_DIR    = os.path.join(ROOT_DIR, "images")
META_DIR   = os.path.join(ROOT_DIR, "meta")
TRAIN_SPL  = os.path.join(META_DIR, "train.txt")
TEST_SPL   = os.path.join(META_DIR, "test.txt")
CLASSES_TXT = os.path.join(META_DIR, "classes.txt")

BATCH_SIZE = 32
IMG_SIZE   = (299, 299)           # InceptionV3 native resolution
NUM_EPOCHS = 50
AUTOTUNE   = tf.data.AUTOTUNE

# ───────────────────── utilities ────────────────────────────
def read_classes(path):
    with open(path) as f:
        classes = [ln.strip() for ln in f]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx

def read_split(path):
    with open(path) as f:
        # Each line like "apple_pie/1005649" (no extension)
        return [ln.strip() for ln in f]

def make_file_label_lists(split_lines, class_to_idx):
    files, labels = [], []
    for rel in split_lines:
        cls, img_id = rel.split("/", 1)
        files.append(os.path.join(IMG_DIR, cls, f"{img_id}.jpg"))
        labels.append(class_to_idx[cls])
    return files, labels

def decode_image(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float16) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    return img, label

def build_dataset(files, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(len(files))
    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ───────────────────── data preparation ─────────────────────
classes, class_to_idx = read_classes(CLASSES_TXT)
train_lines = read_split(TRAIN_SPL)
test_lines  = read_split(TEST_SPL)

train_files, train_labels = make_file_label_lists(train_lines, class_to_idx)
test_files,  test_labels  = make_file_label_lists(test_lines,  class_to_idx)

train_ds = build_dataset(train_files, train_labels, training=True)
val_ds   = build_dataset(test_files,  test_labels,  training=False)

# Save class indices for later inference
with open("class_indices.json", "w") as f:
    json.dump(class_to_idx, f, indent=2)

# ───────────────────── model definition ─────────────────────
base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False          # first train only the head

inputs = base_model.input
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
preds = Dense(
    len(classes),
    activation="softmax",
    dtype="float32"                   # keep final logits in FP32
)(x)

model = tf.keras.Model(inputs, preds)

# ───────────────────── compilation ──────────────────────────
opt = SGD(learning_rate=1e-3, momentum=0.9)
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ───────────────────── callbacks ────────────────────────────
ckpt = ModelCheckpoint(
    "checkpoints/best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

csv_log = CSVLogger("training.log")

lr_plateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

callbacks = [ckpt, csv_log, lr_plateau]

# ───────────────────── training (head) ──────────────────────
model.summary()
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS // 2,
    callbacks=callbacks
)

# ───────────────────── fine-tune whole network ──────────────
base_model.trainable = True
opt_fine = SGD(learning_rate=1e-4, momentum=0.9)
model.compile(
    optimizer=opt_fine,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    initial_epoch=NUM_EPOCHS // 2,
    callbacks=callbacks
)
