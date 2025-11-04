import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# =====================================
# 1️⃣ Load Model
# =====================================
model_path = "ckd_cnn_model_v4.h5"
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded successfully from '{model_path}'")

# =====================================
# 2️⃣ Define Classes (same as training)
# =====================================
class_labels = ["Cyst", "Stone", "Tumour", "Normal"]

# =====================================
# 3️⃣ Set Test Folder
# =====================================
test_dir = r"kidney-dataset/test"  # 👈 Change path if needed

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ Test folder not found at {test_dir}")

# =====================================
# 4️⃣ Predict Images in Folder
# =====================================
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(test_files) == 0:
    raise ValueError(f"❌ No image files found in {test_dir}")

print(f"\n🧾 Found {len(test_files)} images for prediction.\n")

for i, filename in enumerate(test_files):
    img_path = os.path.join(test_dir, filename)
    
    # Load & preprocess
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    pred_class = class_labels[pred_idx]
    
    print(f"[{i+1}/{len(test_files)}] 🩺 {filename} → {pred_class} ({confidence:.2f}%)")

print("\n✅ All predictions completed successfully!")

# =====================================
# 5️⃣ Optional: Show sample predictions
# =====================================
show_count = min(5, len(test_files))
plt.figure(figsize=(12, 6))
for j, filename in enumerate(test_files[:show_count]):
    img_path = os.path.join(test_dir, filename)
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    pred_class = class_labels[pred_idx]
    
    plt.subplot(1, show_count, j + 1)
    plt.imshow(load_img(img_path))
    plt.title(f"{pred_class}\n{confidence:.1f}%")
    plt.axis("off")

plt.tight_layout()
plt.show()
