from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization # type: ignore
from tensorflow.keras.applications import EfficientNetB3 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Constants and Hyperparameters
DATA_DIR = "data"
BATCH_SIZE = 32
IMAGE_SIZE = (300, 300)  # Resize images to (300, 300) for EfficientNetB3
NUM_CLASSES = 5  # Number of classes in the dataset
EPOCHS_BASE = 10  # Initial training epochs (with frozen EfficientNet model)
EPOCHS_FINE = 35  # Fine-tuning epochs
SEED = 123  # Random seed for reproducibility

# Load and preprocess datasets
def load_data():
    """Load the dataset and verify its structure."""
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    verify_class_names(train_ds)
    return train_ds, val_ds

def verify_class_names(train_ds):
    """Verify the class names match the expected labels."""
    class_names = train_ds.class_names
    print(f"Verified Class Names: {class_names}")
    if class_names != ["0", "1", "2", "3", "4"]:
        raise ValueError("Class names in the dataset are not as expected. Please check the data directory structure.")

# Analyze class distribution
def analyze_class_distribution(dataset, dataset_name="Dataset"):
    class_counts = {}
    for _, labels in dataset.unbatch():
        label = labels.numpy()
        class_counts[label] = class_counts.get(label, 0) + 1
    print(f"{dataset_name} Class Distribution: {class_counts}")
    return class_counts

# Validate and preprocess the dataset
def preprocess_dataset(train_ds, val_ds):
    """Preprocess the dataset: normalize and augment."""
    def preprocess(image, label):
        # Normalize images
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(image, label):
        # Apply data augmentation
        image = data_augmentation(image)
        return image, label

    train_ds = train_ds.map(preprocess).map(augment)
    val_ds = val_ds.map(preprocess)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Compute class weights
def compute_class_weights(train_ds):
    """Compute class weights for handling class imbalance."""
    label_list = []
    for _, labels in train_ds.unbatch():
        label_list.append(labels.numpy())  # Append scalar labels
    class_weights = compute_class_weight("balanced", classes=np.unique(label_list), y=label_list)
    debug_class_weights(dict(enumerate(class_weights)))
    return dict(enumerate(class_weights))

def debug_class_weights(class_weights):
    """Debug and log class weights."""
    print(f"Debug: Class Weights = {class_weights}")
    total_weight = sum(class_weights.values())
    print(f"Debug: Total Weight = {total_weight}")
    for class_id, weight in class_weights.items():
        print(f"Class {class_id}: {weight:.4f}")

# Weighted cross-entropy loss function
def weighted_cross_entropy_loss(weights):
    """Weighted cross-entropy loss for imbalanced datasets."""
    def loss_fn(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(weights))  # One-hot encode labels
        weights_tensor = tf.constant(weights, dtype=tf.float32)
        weighted_logits = y_true * tf.math.log(y_pred + 1e-7) * weights_tensor
        return -tf.reduce_mean(tf.reduce_sum(weighted_logits, axis=-1))  # Reduce over the batch
    return loss_fn

# Build the EfficientNetB3 model
def build_model():
    """Build the EfficientNetB3 model with a custom classification head using Intel optimizations."""
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_tensor=Input(shape=(*IMAGE_SIZE, 3)))
    base_model.trainable = False  # Freeze the base model

    # Add a custom classification head
    x = Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)  # Use Intel-optimized Dense layer
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

# Compile and train the model
def compile_and_train(model, train_ds, val_ds, class_weights, epochs, learning_rate):
    """Compile and train the model."""
    weights = [class_weights[i] for i in range(NUM_CLASSES)]
    loss_fn = weighted_cross_entropy_loss(weights)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]
    )
    return history

# Evaluate the model
def evaluate_model(model, val_ds):
    """Evaluate the model and print precision, recall, and F1-score."""
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))  # Convert probabilities to class indices

    print(classification_report(
        y_true, y_pred, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
    ))
    plot_confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"], 
                    yticklabels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"])

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Main function
def main():
    """Run the complete training pipeline."""
    train_ds, val_ds = load_data()
    analyze_class_distribution(train_ds, "Training Dataset")
    analyze_class_distribution(val_ds, "Validation Dataset")

    train_ds, val_ds = preprocess_dataset(train_ds, val_ds)

    # Compute class weights
    class_weights = compute_class_weights(train_ds)

    # Build and train the model
    model, base_model = build_model()

    # Train with frozen base model
    print("Training with frozen base model...")
    compile_and_train(model, train_ds, val_ds, class_weights, EPOCHS_BASE, learning_rate=1e-3)

    # Fine-tune by unfreezing more layers
    print("Fine-tuning the model...")
    for layer in base_model.layers:
        layer.trainable = True
    compile_and_train(model, train_ds, val_ds, class_weights, EPOCHS_FINE, learning_rate=1e-5)

    # Save the final model
    model.save("trained_model.h5")  # Save in TensorFlow SavedModel format

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, val_ds)

if __name__ == "__main__":
    main()
"""

                 precision  recall   f1-score   support

     Class 0       0.58      0.78      0.67      1403
     Class 1       0.27      0.20      0.23       658
     Class 2       0.60      0.26      0.36       896
     Class 3       0.50      0.46      0.48       445
     Class 4       0.27      0.83      0.41       120

    accuracy                           0.50      3522
   macro avg       0.44      0.51      0.43      3522
weighted avg       0.51      0.50      0.48      3522


"""