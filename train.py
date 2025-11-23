import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from collections import Counter
import matplotlib.pyplot as plt

#paths for folders
dataset_dir = "kidney-dataset"
train_dir = os.path.join(dataset_dir, "train")
valid_dir = os.path.join(dataset_dir, "valid")

print("✅ Libraries loaded successfully!")
print(f"📁 Train folder: {train_dir}")
print(f"📁 Valid folder: {valid_dir}")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"❌ Train folder not found at {train_dir}")
if not os.path.exists(valid_dir):
    raise FileNotFoundError(f"❌ Validation folder not found at {valid_dir}")

classes = ["cyst", "stone", "tumour", "normal"]

def get_label_from_name(filename):
    name = filename.lower()
    if "tumor" in name or "tumour" in name:
        return 2
    for idx, c in enumerate(classes):
        if c in name:
            return idx
    return None

def load_images_from_folder(folder):
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n📂 Loading from {folder} ({len(files)} files)...")
    for i, file in enumerate(files):
        label = get_label_from_name(file)
        if label is None:
            continue
        try:
            img_path = os.path.join(folder, file)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label)
            if i % 2000 == 0 and i > 0:
                print(f"  ✅ Loaded {i} images...")
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
    return np.array(X), np.array(y)

#load data
X_train, y_train = load_images_from_folder(train_dir)
X_val, y_val = load_images_from_folder(valid_dir)

print(f"\n✅ Loaded training images: {len(X_train)}")
print(f"✅ Loaded validation images: {len(X_val)}")

print("📊 Train class distribution:", dict(Counter(y_train)))
print("📊 Valid class distribution:", dict(Counter(y_val)))

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(X_train)

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print("⚖️ Class Weights:", class_weights)

#Cnn here
print("\n🏗️ Building CNN model...")
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
print("✅ Model compiled successfully!\n")
model.summary()

#training model
print("\n🚀 Starting model training...")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32),
    validation_data=(X_val, y_val_cat),
    epochs=20,
    class_weight=class_weights
)
print("\n✅ Model training complete!")

model.save("ckd_cnn_model_v4.h5")
print("💾 Model saved as 'ckd_cnn_model_v4.h5'")

#training performance code
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
