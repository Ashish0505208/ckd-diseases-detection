import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


print("🔄 Loading trained model...")
model = tf.keras.models.load_model('ckd_cnn_model_v3.h5')
print("✅ Model loaded successfully!\n")


class_labels = ['Cyst', 'Stone', 'Tumour', 'Normal']
print("🧩 Class labels loaded:", class_labels, "\n")


folder_path = 'kidney-dataset/test'  #folder path
print(f"📁 Loading images from folder: {folder_path}\n")


image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"🧾 Found {len(image_files)} images to predict.\n")

count = 0

for filename in image_files:
    count += 1
    img_path = os.path.join(folder_path, filename)
    
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(prediction) * 100
    
    print(f"[{count}/{len(image_files)}] 🩺 {filename} → {predicted_class} ({confidence:.2f}%)")

print("\n✅ All predictions completed successfully!")