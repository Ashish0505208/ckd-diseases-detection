import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
from collections import Counter
import matplotlib.pyplot as plt

# =====================================
# 1️⃣ Setup Paths
# =====================================
dataset_dir = "kidney-dataset"
train_dir = os.path.join(dataset_dir, "train")
valid_dir = os.path.join(dataset_dir, "valid")

print("✅ Libraries loaded successfully!")

classes = ["cyst", "stone", "tumour", "normal"]

# =====================================
# 2️⃣ Label Extraction Function
# =====================================
def get_label_from_name(filename):
    name = filename.lower()
    if "tumor" in name or "tumour" in name:
        return 2
    for idx, c in enumerate(classes):
        if c in name:
            return idx
    return None

# =====================================
# 3️⃣ Load Dataset Function
# =====================================
def load_images_from_folder(folder):
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n📂 Loading from {folder} ({len(files)} files)...")
    for i, file in enumerate(files):
        label = get_label_from_name(file)
        if label is None:
            continue
        try:
            img = load_img(os.path.join(folder, file), target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label)
            if i % 2000 == 0 and i > 0:
                print(f"✅ Loaded {i} images...")
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
    return np.array(X), np.array(y)

# =====================================
# 4️⃣ Load Train & Validation Data
# =====================================
X_train, y_train = load_images_from_folder(train_dir)
X_val, y_val = load_images_from_folder(valid_dir)

print(f"\n✅ Training images: {len(X_train)}")
print(f"✅ Validation images: {len(X_val)}")

print("📊 Train class distribution:", dict(Counter(y_train)))
print("📊 Valid class distribution:", dict(Counter(y_val)))

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

# =====================================
# 5️⃣ Data Augmentation (stronger)
# =====================================
datagen = ImageDataGenerator(
    rotation_range=35,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.3,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)
datagen.fit(X_train)

# =====================================
# 6️⃣ Compute Class Weights (for balancing)
# =====================================
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print("⚖️ Class Weights:", class_weights)

# =====================================
# 7️⃣ Improved CNN Architecture
# =====================================
print("\n🏗️ Building Improved CNN model (VGG-style)...")

model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation="relu", padding='same', input_shape=(128,128,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    # Block 3
    Conv2D(128, (3,3), activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    # Dense
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ Model compiled successfully!\n")
model.summary()

# =====================================
# 8️⃣ Callbacks
# =====================================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
    ModelCheckpoint('best_ckd_model_v5.h5', monitor='val_accuracy', save_best_only=True)
]

# =====================================
# 9️⃣ Train Model
# =====================================
print("\n🚀 Starting model training...")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32),
    validation_data=(X_val, y_val_cat),
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n✅ Model training complete!")

# =====================================
# 🔟 Save Final Model
# =====================================
model.save('/content/drive/MyDrive/ckd_cnn_model_v5.h5')
print("💾 Model saved as 'ckd_cnn_model_v5_final.h5'")

# =====================================
# 📊 Plot Performance
# =====================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()
