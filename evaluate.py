# evaluate.py

from tensorflow.keras.models import load_model
from utils import setup_generator
from predict import build_label_map
import numpy as np

# Hardcoded parameters
MODEL_PATH = "checkpoints/best.keras"
TRAIN_PATH = "food-101/train"
TEST_PATH  = "food-101/test"
BATCH_SIZE = 32
IMG_DIMS   = (224, 224)  # match training image dimensions

# 1) Load the trained model
model = load_model(MODEL_PATH)

# 2) Build data generators
train_generator, test_generator = setup_generator(
    TRAIN_PATH,
    TEST_PATH,
    BATCH_SIZE,
    IMG_DIMS
)  # setup_generator returns (train_gen, validation_gen) citeturn6file0

# 3) Recreate index->label mapping for predictions
label_map = build_label_map(
    TRAIN_PATH,
    target_size=IMG_DIMS,
    batch_size=BATCH_SIZE
)  # returns {index: label} citeturn6file1

# 4) Predict on the entire test set
predictions = model.predict(test_generator, verbose=1)

# 5) Compute predicted vs. true indices
predicted_indices = np.argmax(predictions, axis=1)
true_indices = test_generator.classes

# 6) Calculate and print accuracy
total = len(true_indices)
correct = int(np.sum(predicted_indices == true_indices))
accuracy = correct / total if total > 0 else 0
print(f"Test accuracy: {correct}/{total} = {accuracy:.2%}")
