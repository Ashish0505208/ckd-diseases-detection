import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, classification_report

test_dir = "kidney-dataset/test"
model_path = "ckd_cnn_81accuracy.h5"

classes = ["cyst", "stone", "tumor", "normal"]

def get_label_from_name(filename):
    name = filename.lower()
    if "tumor" in name:
        return 2
    for idx, c in enumerate(classes):
        if c in name:
            return idx
    return None

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded!")

y_true = []
y_pred = []

files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

print(f"📂 Found {len(files)} test images.\n")

for file in files:
    true_label = get_label_from_name(file)
    if true_label is None:
        print("⚠️ Skipped unlabeled:", file)
        continue

    path = os.path.join(test_dir, file)

    try:
        img = load_img(path, target_size=(128, 128))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
    except:
        print("⚠️ Error loading:", file)
        continue

    pred = model.predict(arr, verbose=0)
    pred_class = np.argmax(pred)

    y_true.append(true_label)
    y_pred.append(pred_class)

print("\n🎯 Accuracy:", accuracy_score(y_true, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))