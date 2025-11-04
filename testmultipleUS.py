import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# =======================================
# 1️⃣ Load the trained model
# =======================================
print("🔄 Loading trained model...")
model = tf.keras.models.load_model('ckd_cnn_model_v3.h5')
print("✅ Model loaded successfully!\n")

# =======================================
# 2️⃣ Define class labels (must match training)
# =======================================
class_labels = ['Cyst', 'Stone', 'Tumour', 'Normal']
print("🧩 Class labels loaded:", class_labels, "\n")

# =======================================
# 3️⃣ Set your folder path (test images)
# =======================================
folder_path = 'kidney-dataset/test'  # 👈 Change path if needed
print(f"📁 Loading images from folder: {folder_path}\n")

# =======================================
# 4️⃣ Loop through all images in the folder
# =======================================
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"🧾 Found {len(image_files)} images to predict.\n")

# Counter
count = 0

for filename in image_files:
    count += 1
    img_path = os.path.join(folder_path, filename)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(prediction) * 100
    
    # Print progress every 100 images or every file
    print(f"[{count}/{len(image_files)}] 🩺 {filename} → {predicted_class} ({confidence:.2f}%)")

print("\n✅ All predictions completed successfully!")

# =======================================
# 5️⃣ Optional: Show a few sample predictions
# =======================================
show_samples = True
if show_samples:
    print("\n🖼️ Displaying a few sample predictions...\n")
    sample_images = image_files[:5]  # show first 5
    plt.figure(figsize=(12, 6))
    for i, filename in enumerate(sample_images):
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = np.max(prediction) * 100
        
        plt.subplot(1, 5, i+1)
        plt.imshow(image.load_img(img_path))
        plt.title(f"{predicted_class}\n{confidence:.1f}%")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
