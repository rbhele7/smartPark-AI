import os
import json
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

# IMPORTANT:
# Small number so tuning finishes quickly
TRAIN_SAMPLES = 3000
VALID_SAMPLES = 600

# Only 2 epochs per experiment
EPOCHS = 2

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not os.path.exists(VALID_DIR):
    raise FileNotFoundError(
        f"Validation directory not found: {VALID_DIR}"
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=True,

    seed=42
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

valid_dataset = tf.keras.utils.image_dataset_from_directory(

    VALID_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)


# ============================================================
# PRINT CLASSES
# ============================================================

print("\nClasses:")

print(
    train_dataset.class_names
)


# ============================================================
# TAKE SMALL SUBSET
# ============================================================

print("\nUsing a small subset for FAST hyperparameter tuning.")

print(
    f"Training samples: approximately {TRAIN_SAMPLES}"
)

print(
    f"Validation samples: approximately {VALID_SAMPLES}"
)


train_batches = TRAIN_SAMPLES // BATCH_SIZE

valid_batches = VALID_SAMPLES // BATCH_SIZE


train_dataset = train_dataset.take(
    train_batches
)

valid_dataset = valid_dataset.take(
    valid_batches
)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    learning_rate,
    dropout
):

    print(
        f"\nCreating model:"
        f"\nLearning Rate = {learning_rate}"
        f"\nDropout = {dropout}"
    )


    # --------------------------------------------------------
    # MobileNetV2
    # --------------------------------------------------------

    # IMPORTANT:
    # weights=None prevents the SSL download problem
    # you previously encountered on your Mac.
    #
    # The tuning is only being used to demonstrate
    # hyperparameter selection quickly.

    base_model = MobileNetV2(

        input_shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        ),

        include_top=False,

        weights=None
    )


    # Freeze base model

    base_model.trainable = False


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = layers.Input(

        shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        )
    )


    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    x = layers.RandomFlip(
        "horizontal"
    )(inputs)


    x = layers.RandomRotation(
        0.05
    )(x)


    x = layers.RandomZoom(
        0.1
    )(x)


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        x
    )


    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    x = base_model(

        x,

        training=False
    )


    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)


    x = layers.Dense(
        128,
        activation="relu"
    )(x)


    x = layers.Dropout(
        dropout
    )(x)


    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = Model(
        inputs,
        outputs
    )


    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        optimizer=Adam(
            learning_rate=learning_rate
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )


    return model


# ============================================================
# HYPERPARAMETERS
# ============================================================

experiments = [

    {
        "name": "Experiment_1",
        "learning_rate": 0.001,
        "dropout": 0.3
    },

    {
        "name": "Experiment_2",
        "learning_rate": 0.0001,
        "dropout": 0.3
    },

    {
        "name": "Experiment_3",
        "learning_rate": 0.0001,
        "dropout": 0.5
    }

]


# ============================================================
# STORE RESULTS
# ============================================================

results = []


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for experiment in experiments:

    print("\n")
    print("=" * 65)

    print(
        experiment["name"]
    )

    print("=" * 65)

    learning_rate = experiment[
        "learning_rate"
    ]

    dropout = experiment[
        "dropout"
    ]


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(

        learning_rate=learning_rate,

        dropout=dropout
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(

        train_dataset,

        validation_data=valid_dataset,

        epochs=EPOCHS,

        verbose=1
    )


    # --------------------------------------------------------
    # Get best metrics
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history[
            "val_accuracy"
        ]
    )


    best_val_auc = max(
        history.history[
            "val_auc"
        ]
    )


    best_val_loss = min(
        history.history[
            "val_loss"
        ]
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "experiment":
            experiment["name"],

        "learning_rate":
            learning_rate,

        "dropout":
            dropout,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "training_samples":
            TRAIN_SAMPLES,

        "validation_samples":
            VALID_SAMPLES,

        "best_validation_accuracy":
            float(
                best_val_accuracy
            ),

        "best_validation_auc":
            float(
                best_val_auc
            ),

        "best_validation_loss":
            float(
                best_val_loss
            )
    }


    results.append(
        result
    )


    print("\nExperiment Result:")

    print(
        f"Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Validation AUC: "
        f"{best_val_auc * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{best_val_loss:.4f}"
    )


# ============================================================
# SORT BY VALIDATION ACCURACY
# ============================================================

results = sorted(

    results,

    key=lambda x:
        x["best_validation_accuracy"],

    reverse=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(

    OUTPUT_DIR,

    "hyperparameter_results.json"
)


with open(
    result_file,
    "w"
) as file:

    json.dump(

        results,

        file,

        indent=4
    )


# ============================================================
# BEST EXPERIMENT
# ============================================================

best = results[0]


print("\n")
print("=" * 65)

print("BEST HYPERPARAMETERS")

print("=" * 65)

print(
    f"Learning Rate : "
    f"{best['learning_rate']}"
)

print(
    f"Dropout       : "
    f"{best['dropout']}"
)

print(
    f"Batch Size    : "
    f"{best['batch_size']}"
)

print(
    f"Epochs        : "
    f"{best['epochs']}"
)

print(
    f"Validation Accuracy : "
    f"{best['best_validation_accuracy'] * 100:.2f}%"
)

print(
    f"Validation AUC      : "
    f"{best['best_validation_auc'] * 100:.2f}%"
)

print(
    f"Validation Loss     : "
    f"{best['best_validation_loss']:.4f}"
)

print("=" * 65)


print(
    f"\nResults saved to:"
    f" {result_file}"
)