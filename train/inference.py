from diffusers import StableDiffusionPipeline
import torch
import os

validation_prompts = [
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


# Define paths
RANK = 128
PRETRAINED_MODEL_PATH = "../stable-diffusion-2-1"
FINETUNED_MODEL_PATH = f"../output/fine_tuned_model_{RANK}/pytorch_lora_weights.safetensors"
NUM_IMAGES = 1  # Number of images to generate
LORA_WEIGHT = True

for PROMPT in validation_prompts:
    OUTPUT_IMAGE_DIR = f"../test/figure/nobel_{RANK}_inference/" + PROMPT.split(",")[0].lower().replace(" ","_")

    # Ensure output directory exists
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_PATH, torch_dtype=torch.float16
    )
    
    if LORA_WEIGHT:
        pipe.load_lora_weights(FINETUNED_MODEL_PATH)
    pipe.to("cuda")
    
    # Generate images
    generator = torch.manual_seed(42)
    for i in range(NUM_IMAGES):
        image = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]
        output_path = os.path.join(OUTPUT_IMAGE_DIR, f"nobel_{i+1}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")
