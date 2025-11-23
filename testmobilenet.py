import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


model_path = "MobileNetV2_Kidney_Final.h5"
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

class_labels = ["Cyst", "Stone", "Tumour", "Normal"]

test_folder = r"kidney-dataset/test"

if not os.path.exists(test_folder):
    raise FileNotFoundError(f"❌ Folder not found: {test_folder}")

files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(files) == 0:
    raise ValueError("❌ No images found in test folder!")

print(f"\n🔍 Found {len(files)} test images.\n")


for idx, filename in enumerate(files, start=1):
    img_path = os.path.join(test_folder, filename)

    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr, verbose=0)
    pred_idx = np.argmax(preds)
    confidence = preds[0][pred_idx] * 100

    print(f"[{idx}/{len(files)}] {filename} → {class_labels[pred_idx]} ({confidence:.2f}%)")

print("\n✅ All predictions complete!")