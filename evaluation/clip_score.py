import os
from PIL import Image
import torch
from clip import load, tokenize
import numpy as np

# Define paths
RANK = 128
image_folders_path = f"../test/figure/nobel_{RANK}_inference"  # Path to the image folders
captions = [
    "Donald Trump with a warm smile, and wearing a scarf, in the style of Nobel Laureate.",
    "Donald Trump with a serious face, and wearing a hat, in the style of Nobel Laureate.",
    "A cheerful woman with curly hair, a warm smile, and a stylish scarf, in the style of Nobel Laureate.",
    "A wise man with glasses, slightly tousled hair, and a gentle smile, exuding warmth and intellect, in the style of Nobel Laureate.",
    "Elon Musk with a bow tie, and a short hairstyle, in the style of Nobel Laureate.",
    "A poised woman with flowing hair, a calm expression, and an air of elegance, in the style of Nobel Laureate.",
    "A smiling man with a bald head, wearing a sweater and tie, exuding warmth and approachability, in the style of Nobel Laureate.",
    "Leo Messi with an angry expression, and wearing his jersey, in the style of Nobel Laureate.",
    "Justin Bieber feeling shock, exuding warmth and approachability, in the style of Nobel Laureate.",
]

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load("ViT-B/32", device=device)
print(device)

# Function to calculate CLIP score for a single image-caption pair
def calculate_clip_score(image_path, caption):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenize([caption]).to(device)
    
    # Get features from the model
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity
        similarity = (image_features @ text_features.T).item()
    return similarity

# Match images with captions and calculate CLIP scores
results = []
for caption in captions:
    # Extract the folder name from the caption prefix
    folder_name = caption.split(",")[0].lower().replace(" ", "_")
    folder_path = os.path.join(image_folders_path, folder_name)
    
    # Check if the folder exists
    if os.path.isdir(folder_path):
        # Pick the first image in the folder
        images = os.listdir(folder_path)
        if images:
            image_path = os.path.join(folder_path, images[0])
            # Calculate CLIP score
            score = calculate_clip_score(image_path, caption)
            results.append((image_path, caption, score))
            print(f"Image: {image_path}, Caption: {caption}, CLIP Score: {score:.4f}")
    else:
        print(f"Folder not found: {folder_path}")

# Save results to a file
clip_scores = []
# output_path = "clip_scores.txt"
# with open(output_path, "w") as f:
for image_path, caption, score in results:
    clip_scores.append(score)
        # f.write(f"Image: {image_path}\nCaption: {caption}\nCLIP Score: {score:.4f}\n\n")

avg_clip = np.mean(np.array(clip_scores))
print(f"\nAverage CLIP Scores: {avg_clip}")
