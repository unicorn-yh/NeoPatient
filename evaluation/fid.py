import os
from pytorch_fid import fid_score
import shutil
from PIL import Image
import torch

def combine_generated_data(data_path, new_path):
    os.makedirs(new_path, exist_ok=True)
    folders = os.listdir(data_path)
    image_counter = 1

    for folder in folders:
        cur_path = os.path.join(data_path, folder)
        if os.path.isdir(cur_path):
            images = os.listdir(cur_path)
            for image in images:
                cur_path2 = os.path.join(cur_path, image)
                if os.path.isfile(cur_path2):
                    new_image_name = f"image_{image_counter:05d}.png"
                    new_image_path = os.path.join(new_path, new_image_name)
                    shutil.copy(cur_path2, new_image_path)
                    image_counter += 1

def validate_and_resize_images(folder_path, size=(299, 299)):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")  # Ensure 3 channels
                img = img.resize(size)  # Resize to the required size
                img.save(file_path)  # Save the resized image
        except Exception as e:
            print(f"Failed to process image: {file_path}, Error: {e}")
            os.remove(file_path)  # Optionally remove invalid files

# Paths
RANK = 128
data_path = f"../test/figure/nobel_{RANK}_inference"
new_path = "../test/figure/combined_images"
real_images_path = "../dataset/nobel/images"

# Combine and process generated images
combine_generated_data(data_path, new_path)

# Validate and resize images
validate_and_resize_images(real_images_path)
validate_and_resize_images(new_path)

# Compute FID
fid_value = fid_score.calculate_fid_given_paths(
    [real_images_path, new_path],
    batch_size=5,  # Adjust based on GPU/CPU memory
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dims=2048  # Dimensionality of features from Inception model
)

print(f"FID: {fid_value}")
