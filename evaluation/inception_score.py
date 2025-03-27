from torchvision.models import inception_v3
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import torch
import numpy as np
import os
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
                    
                    
def calculate_inception_score(image_folder, batch_size=1, splits=3):
    # Load pretrained Inception model
    model = inception_v3(pretrained=True, transform_input=False).eval()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load images
    images = sorted(os.listdir(image_folder))
    img_tensors = []
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        img_tensors.append(transform(image))
    img_tensors = torch.stack(img_tensors)

    # Compute predictions in batches
    preds = []
    with torch.no_grad():
        for i in range(0, len(img_tensors), batch_size):
            batch = img_tensors[i:i+batch_size]
            batch_preds = F.softmax(model(batch), dim=1).cpu().numpy()
            preds.append(batch_preds)
    preds = np.concatenate(preds, axis=0)

    # Compute Inception Score
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = [np.sum(p * np.log(p / py)) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# Paths
RANK = 128
data_path = f"../test/figure/nobel_{RANK}_inference"
new_path = "../test/figure/combined_images"
real_images_path = "../dataset/nobel/images"

# Combine and process generated images
combine_generated_data(data_path, new_path)

print(f"RANK: {RANK}")
inception_mean, inception_std = calculate_inception_score(new_path)
print(f"Inception Score: {inception_mean:.2f} ± {inception_std:.2f}")
