import os
import shutil

base = "kidney-dataset/train"
classes = ["cyst", "stone", "tumour", "normal"]

# Create a temporary folder to hold unsorted images
temp_folder = os.path.join(base, "_unsorted")
os.makedirs(temp_folder, exist_ok=True)

# Move images out of class folders
for c in classes:
    class_path = os.path.join(base, c)
    
    if not os.path.exists(class_path):
        continue

    for file in os.listdir(class_path):
        src = os.path.join(class_path, file)
        dst = os.path.join(temp_folder, file)

        # Only move files, skip nested folders
        if os.path.isfile(src):
            shutil.move(src, dst)

# Move all images back into the main valid folder
for file in os.listdir(temp_folder):
    src = os.path.join(temp_folder, file)
    dst = os.path.join(base, file)
    shutil.move(src, dst)

# Remove temp folder and empty class folders
shutil.rmtree(temp_folder)

for c in classes:
    class_path = os.path.join(base, c)
    if os.path.isdir(class_path) and len(os.listdir(class_path)) == 0:
        os.rmdir(class_path)

print("Undo complete! All images moved back to 'valid' and class folders removed.")
