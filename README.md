# Food Nutrition Estimator

A complete, end‑to‑end workflow for training, evaluating, and using an InceptionV3 classifier on the [Food‑101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

---

## 1. Environment setup

> **Python 3.10+ is recommended.**

```bash
# clone this repo and enter it
$ git clone https://github.com/jengalbert2022/FoodNutritionEstimator.git
$ cd food101

# create and activate a virtual environment
$ python3 -m venv food101-env
$ source food101-env/bin/activate
```

| Use‑case              | Pip command                                                                    |
| --------------------- | ------------------------------------------------------------------------------ |
| **CPU‑only**          | `pip install tensorflow tensorflow_datasets matplotlib pandas scipy`           |
| **NVIDIA GPU (CUDA)** | `pip install tensorflow[and-cuda] tensorflow_datasets matplotlib pandas scipy` |

---

## 2. Download and prepare Food‑101 images

```bash
$ wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
$ tar -xvf food-101.tar.gz

# split the official images into ./food-101/train and ./food-101/test
$ python split_train_test_folder.py
```

After the split the folder tree should look like this:

```
food-101/
 ├── train/
 │    ├── apple_pie/XXX.jpg
 │    └── …
 ├── test/
 │    └── …
 └── meta/
      ├── classes.txt
      ├── train.txt  # original file list
      └── test.txt
```

---

## 3. Training a model

```bash
$ python train.py
```

* Two training phases are run: first the classification head only, then the entire network is fine‑tuned.
* Checkpoints are written to `; the best model is saved as `.
* `class_indices.json` is written at the repo root for later inference.

---

## 4. Evaluating accuracy

```bash
$ python evaluate.py              # uses checkpoints/best.keras by default
```

`evaluate.py` will

1. load the best checkpoint,
2. create generators with **`utils.setup_generator`**, and
3. print overall top‑1 accuracy on *food‑101/test*.

Change paths at the top of the script if your layout differs.

---

## 5. Predicting on sample images

```bash
# single image
$ python predict.py \
    --model checkpoints/best.keras \
    --train-dir food-101/train \
    gyoza.jpg

# multiple images, custom output file
$ python predict.py -m checkpoints/best.keras -t food-101/train \
    -o my_preds.txt gyoza.jpg wings.jpg nachos.jpg
```

* Top‑5 predictions and confidences are appended to **`my_preds.txt`**.
* `--train-dir` must point to the **train** folder created by `split_train_test_folder.py` so that label indices match.

---
