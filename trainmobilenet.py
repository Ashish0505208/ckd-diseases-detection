import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ================================
# 1️⃣ CLASS LABELS (MATCH CNN ORDER)
# ================================
classes = ["cyst", "stone", "tumor", "normal"]
num_classes = len(classes)

def get_label_from_name(filename):
    fname = filename.lower()
    if "tumor" in fname: return 2
    for idx, c in enumerate(classes):
        if c in fname: return idx
    return None

# ================================
# 2️⃣ CUSTOM STREAMING GENERATOR
# ================================
class KidneyGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder, batch_size=32, img_size=(224,224)):
        self.folder = folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and get_label_from_name(f) is not None
        ]

    def __len__(self):
        return len(self.files) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.files[idx*self.batch_size : (idx+1)*self.batch_size]
        X, y = [], []

        for f in batch_files:
            path = os.path.join(self.folder, f)
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img)
            img = mobilenet_v2.preprocess_input(img)

            label = get_label_from_name(f)
            X.append(img)
            y.append(label)

        return np.array(X), tf.keras.utils.to_categorical(y, num_classes)

# ================================
# 3️⃣ LOAD TRAIN + VALID GENERATORS
# ================================
train_dir = "kidney-dataset/train"
valid_dir = "kidney-dataset/valid"

train_gen = KidneyGenerator(train_dir, batch_size=32)
valid_gen = KidneyGenerator(valid_dir, batch_size=32)

print("Train samples:", len(train_gen.files))
print("Valid samples:", len(valid_gen.files))

# ================================
# 4️⃣ BUILD MOBILE-NET MODEL
# ================================
base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================
# 5️⃣ TRAIN MODEL
# ================================
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20
)

model.save("MobileNetV2_Kidney_Final.h5")
print("Saved as MobileNetV2_Kidney_Final.h5")

# ================================
# 6️⃣ PLOT ACCURACY + LOSS
# ================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")

plt.show()
