import os
import json
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

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

            # Preprocess & predict
            x     = preprocess_image(img_path)
            preds = model.predict(x)[0]  # shape = (num_classes,)

            # Get top-5 predictions
            top5 = preds.argsort()[-5:][::-1]

            f.write(f"\nImage: {img_path}\n")
            for idx in top5:
                label = labels.get(idx, f"#{idx}")
                prob  = preds[idx] * 100
                f.write(f"  {label:20s}: {prob:5.2f}%\n")

    print(f"Predictions written to {args.output_file}")

if __name__ == "__main__":
    main()