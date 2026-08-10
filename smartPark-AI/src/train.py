import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "parking_cnn.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "parking_cnn_final.keras"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

required_directories = [
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,

    os.path.join(TRAIN_DIR, "occupied"),
    os.path.join(TRAIN_DIR, "vacant"),

    os.path.join(VALID_DIR, "occupied"),
    os.path.join(VALID_DIR, "vacant"),

    os.path.join(TEST_DIR, "occupied"),
    os.path.join(TEST_DIR, "vacant"),
]

print("\nChecking dataset directories...\n")

for directory in required_directories:

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"\nDataset directory not found:\n"
            f"{directory}\n\n"
            f"Run:\n"
            f"python src/prepare_dataset.py\n"
            f"first."
        )

    print(f"OK: {directory}")


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(directory):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    count = 0

    for filename in os.listdir(directory):

        if filename.lower().endswith(extensions):

            count += 1

    return count


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

for split in ["train", "valid", "test"]:

    occupied_dir = os.path.join(
        "dataset",
        split,
        "occupied"
    )

    vacant_dir = os.path.join(
        "dataset",
        split,
        "vacant"
    )

    occupied_count = count_images(
        occupied_dir
    )

    vacant_count = count_images(
        vacant_dir
    )

    total = occupied_count + vacant_count

    print(f"\n{split.upper()}")

    print(
        f"  Occupied : {occupied_count}"
    )

    print(
        f"  Vacant   : {vacant_count}"
    )

    print(
        f"  Total    : {total}"
    )


# ============================================================
# LOAD TRAIN DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=True,

    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

valid_dataset = tf.keras.utils.image_dataset_from_directory(

    VALID_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)


# ============================================================
# CHECK CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClass names:")
print(class_names)


# Keras sorts directory names alphabetically.
#
# occupied -> label 0
# vacant   -> label 1
#
# This is important for prediction.py.

if class_names != ["occupied", "vacant"]:

    raise ValueError(
        "\nUnexpected class names.\n"
        f"Found: {class_names}\n"
        "Expected: ['occupied', 'vacant']"
    )


# Save class names
with open(
    os.path.join(
        MODEL_DIR,
        "class_names.json"
    ),
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


# ============================================================
# OPTIMIZE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

train_occupied = count_images(
    os.path.join(
        TRAIN_DIR,
        "occupied"
    )
)

train_vacant = count_images(
    os.path.join(
        TRAIN_DIR,
        "vacant"
    )
)

y_train = np.array(
    [0] * train_occupied
    +
    [1] * train_vacant
)


classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        classes,
        weights
    )
}

print("\nClass weights:")

print(
    f"Occupied (0): "
    f"{class_weights[0]:.4f}"
)

print(
    f"Vacant (1): "
    f"{class_weights[1]:.4f}"
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            0.10
        )

    ],

    name="data_augmentation"
)


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(

    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet"
)


# ============================================================
# FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(

    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),

    name="parking_image"
)


# Data augmentation
x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


# Feature extraction
x = base_model(
    x,
    training=False
)


# Reduce feature maps
x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


# Dense layer
x = layers.Dense(
    128,
    activation="relu",
    name="dense_features"
)(x)


# Regularization
x = layers.Dropout(
    0.4,
    name="dropout"
)(x)


# Binary output
outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="occupancy_output"
)(x)


# Create model
model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="SmartParkingCNN"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )

    ]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    filepath=BEST_MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    mode="min",

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


csv_logger = CSVLogger(

    HISTORY_PATH,

    append=False
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[

        checkpoint,

        early_stopping,

        reduce_lr,

        csv_logger

    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_results = best_model.evaluate(

    test_dataset,

    verbose=1,

    return_dict=True
)


print("\nTest Results:")

for metric_name, value in test_results.items():

    print(
        f"{metric_name:12s}: "
        f"{value:.4f}"
    )


# ============================================================
# SAVE TEST RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "test_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        {
            key: float(value)
            for key, value
            in test_results.items()
        },
        f,
        indent=4
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"\nBest model:"
    f"\n{BEST_MODEL_PATH}"
)

print(
    f"\nFinal model:"
    f"\n{FINAL_MODEL_PATH}"
)

print(
    f"\nTraining history:"
    f"\n{HISTORY_PATH}"
)

print(
    f"\nTest results:"
    f"\n{results_path}"
)

print("\nClass mapping:")

print(
    "0 = OCCUPIED"
)

print(
    "1 = VACANT"
)

print("\nYou can now run:")

print(
    "python src/evaluate.py"
)