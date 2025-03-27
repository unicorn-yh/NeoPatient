import lpips
import torch
from PIL import Image
import os
from torchvision import transforms
import shutil

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
                    
# Paths
RANK = 128
data_path = f"../test/figure/nobel_{RANK}_inference"
generated_images_path = "../test/figure/combined_images"
real_images_path = "../dataset/nobel/images"

# Combine and process generated images
combine_generated_data(data_path, generated_images_path)

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg', 'squeeze'

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to match Inception model size
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Iterate through generated images and compute LPIPS
lpips_scores = []
generated_images = sorted(os.listdir(generated_images_path))
real_images = sorted(os.listdir(real_images_path))

for gen_image_name, real_image_name in zip(generated_images, real_images):
    gen_image_path = os.path.join(generated_images_path, gen_image_name)
    real_image_path = os.path.join(real_images_path, real_image_name)
    
    # Load and preprocess images
    gen_image = preprocess(Image.open(gen_image_path).convert("RGB")).unsqueeze(0)  # Add batch dimension
    real_image = preprocess(Image.open(real_image_path).convert("RGB")).unsqueeze(0)
    
    # Compute LPIPS score
    score = lpips_model(gen_image, real_image)
    lpips_scores.append(score.item())

# Average LPIPS score
print(f"RANK: {RANK}")
average_lpips = sum(lpips_scores) / len(lpips_scores)
print(f"Average LPIPS Score: {average_lpips}")
