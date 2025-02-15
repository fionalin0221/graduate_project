import os
import glob

# Specify the directory containing images
image_dir = "/workspace/Data/Datas/CC_Patch"

# Specify image extensions to delete (e.g., PNG, JPG)
extensions = ["*.tif"]

# Loop through each extension and delete matching files
for ext in extensions:
    for image_path in glob.glob(os.path.join(image_dir, ext)):
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")

print("Image deletion completed.")
