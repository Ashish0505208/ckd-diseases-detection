import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

#trained model path
model_path = "ckd_cnn_model_v4.h5"
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded successfully from '{model_path}'")

#classes
class_labels = ["Cyst", "Stone", "Tumour", "Normal"]


test_dir = r"kidney-dataset/test"

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ Test folder not found at {test_dir}")

test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(test_files) == 0:
    raise ValueError(f"❌ No image files found in {test_dir}")

print(f"\n🧾 Found {len(test_files)} images for prediction.\n")

for i, filename in enumerate(test_files):
    img_path = os.path.join(test_dir, filename)
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    pred_class = class_labels[pred_idx]
    
    print(f"[{i+1}/{len(test_files)}] 🩺 {filename} → {pred_class} ({confidence:.2f}%)")

print("\n✅ All predictions completed successfully!")