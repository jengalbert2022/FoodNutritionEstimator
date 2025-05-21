import os
import json
import argparse
import requests
import numpy as np
from functools import lru_cache
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)


USDA_API_KEY = "sinnv6OHnqg26bIlTQwp0Maf5InFD1AM9LPdwaE2"
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_DETAILS_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"



def build_label_map(train_dir, target_size=(224,224), batch_size=32):
    """
    Recreate the ImageDataGenerator used in training to get class_indices,
    then invert it to index->label.
    """
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    # generator.class_indices: dict label->index
    return {v: k for k, v in generator.class_indices.items()}


def preprocess_image(img_path, target_size=(299,299)):
    """
    Load an image file and preprocess it for model.predict().
    """
    img = load_img(img_path, target_size=target_size)
    x   = img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


@lru_cache(maxsize=None)
def fetch_nutrition(label, api_key, max_nutrients=10):
    """
    Return a formatted multiline string of nutrition facts for *label*.
    Results are cached so repeated calls for the same label reuse the HTTP responses.
    """
    query = label.replace("_", " ")

    try:
        # 1) Search for the food
        search_resp = requests.get(
            USDA_SEARCH_URL,
            params={"api_key": api_key, "query": query, "pageSize": 1},
            timeout=10,
        )
        search_resp.raise_for_status()
        foods = search_resp.json().get("foods", [])
        if not foods:
            return "Nutrition data not found."

        fdc_id = foods[0]["fdcId"]

        # 2) Fetch detailed nutrients
        details_resp = requests.get(
            USDA_DETAILS_URL.format(fdc_id=fdc_id),
            params={"api_key": api_key},
            timeout=10,
        )
        details_resp.raise_for_status()
        details = details_resp.json()

        # Build pretty output
        lines = [details.get("description", query)]
        for nutrient in details.get("foodNutrients", [])[:max_nutrients]:
            n = nutrient["nutrient"]
            val = nutrient.get("amount", "N/A")
            unit = n.get("unitName", "")
            lines.append(f"{n['name']}: {val} {unit}")

        return "\n".join(lines)

    except Exception as exc:
        return f"[!] Error fetching nutrition for {label}: {exc}"



def main():
    parser = argparse.ArgumentParser(
        description="Load trained Food-101 model and predict food labels."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to your .keras model file (e.g. model.02-0.95.keras)"
    )
    parser.add_argument(
        "--train-dir", "-t",
        default="food-101/train",
        help="Path to your train/ folder used in training (for class_indices)"
    )
    parser.add_argument(
        "--output-file", "-o",
        default="my_preds.txt",
        help="File to write the predictions (food labels and percentages)"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image file paths to classify"
    )
    args = parser.parse_args()

    # 1) Build index->label map
    labels = build_label_map(args.train_dir)

    # 2) Load your trained model
    model = load_model(args.model)

    # 3) Open the output file and write predictions
    with open(args.output_file, 'w') as f:
        for img_path in args.images:
            if not os.path.isfile(img_path):
                f.write(f"[!] File not found: {img_path}\n")
                continue

             # Predict probabilities
            preds = model.predict(preprocess_image(img_path))[0]

            # Top-5 indices (sorted high→low)
            top5_idx = preds.argsort()[-5:][::-1]

            # Preserve the top-1 label for nutrition lookup and later printing
            top1_idx = int(top5_idx[0])
            top1_label = labels.get(top1_idx, f"#{top1_idx}")

            # Nutrition facts for the top hit
            nutrition_txt = fetch_nutrition(top1_label, USDA_API_KEY)

            # Write results
            f.write(f"\nImage: {img_path}\n")
            for idx in top5_idx:
                lbl = labels.get(idx, f"#{idx}")
                prob = preds[idx] * 100
                f.write(f"  {lbl:20s}: {prob:5.2f}%\n")

            # Append “top1_label” followed by its nutrition block
            f.write(f"\n{nutrition_txt}\n\n")

    print(f"Predictions written to {args.output_file}")

if __name__ == "__main__":
    main()
