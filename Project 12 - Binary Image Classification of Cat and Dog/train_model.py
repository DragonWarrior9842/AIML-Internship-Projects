import os
import shutil
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration (edit these or override via CLI args)
# ---------------------------------------------------------
TRAIN_DATA_DIR = "data/train"
VALIDATION_DATA_DIR = "data/validation"  # separate held-out folder
MODEL_SAVE_PATH = "binary_image_classifier.h5"

IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # cats vs dogs


# ---------------------------------------------------------
# Step 1: clean up stray .ipynb_checkpoints folders
# (present in the original notebook — kept here because messy
#  checkpoint folders inside a class directory will get picked up
#  by flow_from_directory as a bogus extra "class")
# ---------------------------------------------------------
def clean_checkpoints(base_dir: str) -> None:
    for root, dirs, _ in os.walk(base_dir):
        if ".ipynb_checkpoints" in dirs:
            checkpoint_folder = os.path.join(root, ".ipynb_checkpoints")
            shutil.rmtree(checkpoint_folder)
            print(f"Deleted {checkpoint_folder}")
    print("Checkpoint cleanup complete.")


# ---------------------------------------------------------
# Step 2: model architecture (identical to the notebook)
# ---------------------------------------------------------
def create_model() -> tf.keras.Model:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------
# Step 3: data generators (identical augmentation settings)
# ---------------------------------------------------------
def build_generators(train_dir: str, val_dir: str):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    return train_generator, validation_generator


# ---------------------------------------------------------
# Step 4: training routine
# ---------------------------------------------------------
def train():
    if not os.path.isdir(TRAIN_DATA_DIR):
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DATA_DIR}\n"
            "Expected structure: data/train/cats/*.jpg, data/train/dogs/*.jpg"
        )
    if not os.path.isdir(VALIDATION_DATA_DIR):
        raise FileNotFoundError(
            f"Validation folder not found: {VALIDATION_DATA_DIR}\n"
            "Expected structure: data/validation/cats/*.jpg, "
            "data/validation/dogs/*.jpg"
        )

    clean_checkpoints(TRAIN_DATA_DIR)
    clean_checkpoints(VALIDATION_DATA_DIR)

    train_generator, validation_generator = build_generators(
        TRAIN_DATA_DIR, VALIDATION_DATA_DIR
    )

    print("Class indices:", train_generator.class_indices)

    model = create_model()
    model.summary()

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Training and validation completed! Model saved to {MODEL_SAVE_PATH}")


# ---------------------------------------------------------
# Step 5: single-image prediction helper (same logic as the
# notebook's inference cells), usable as a quick sanity check
# ---------------------------------------------------------
def predict_single_image(img_path: str, model_path: str = MODEL_SAVE_PATH,
                         show: bool = True) -> str:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    model = load_model(model_path)

    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    predictions = model.predict(x)
    pred = predictions[0][0]
    print("Raw prediction:", predictions, predictions.shape)

    # class_mode='binary' with flow_from_directory assigns classes
    # alphabetically -> cats=0, dogs=1
    result = "dog" if pred >= 0.5 else "cat"

    if show:
        plt.imshow(img)
        plt.title(f"Prediction: {result}")
        plt.axis("off")
        plt.show()

    return result


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or test the cat/dog CNN classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the model and save it to disk.")

    predict_parser = subparsers.add_parser(
        "predict", help="Run a single-image prediction using a saved model."
    )
    predict_parser.add_argument("image_path", type=str,
                                help="Path to the image to classify.")
    predict_parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH,
                                help="Path to the saved .h5 model.")
    predict_parser.add_argument("--no-show", action="store_true",
                                help="Don't open a matplotlib window.")

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "predict":
        label = predict_single_image(
            args.image_path, model_path=args.model, show=not args.no_show
        )
        print(f"Predicted class: {label}")