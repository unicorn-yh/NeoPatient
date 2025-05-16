#!/bin/bash

# Arguments for the inference script
USE_LORA_WEIGHT=True # Use LoRA weight for finetuned model inference, and do not use it for pretrained model inference.
PRETRAINED_MODEL_PATH="/disks/disk5/private/liyonghui/stable-diffusion-2-1" # Path to the pretrained model
TEST_IMAGE_PATH="../dataset/test" # Path of the test image dataset
MODEL="diffusion"

# Adjust the arguments below if USE_LORA_WEIGHT=True
TEST_MODEL_RANK=8  # Rank for LoRA configuration
TEST_MODEL_DATA=20 # Training data size
FINETUNED_MODEL_PATH="../output/$MODEL/modelckpt_rank${TEST_MODEL_RANK}_data${TEST_MODEL_DATA}/pytorch_lora_weights.safetensors" # Path to finetuned model
SAVE_FIG_PATH="./figure/$MODEL/rank${TEST_MODEL_RANK}_data${TEST_MODEL_DATA}" # Path to save the generated inference image
NUM_IMAGES=1 # Number of images to generate
NUM_INFERENCE_STEPS=30  # Total denoising steps during inference
GUIDANCE_SCALE=7.5  # Controls prompt adherence, higher means more guidance
SEED=42 # Random seed for reproducibility


# Execute the inference script
python inference.py \
  --use_lora_weight $USE_LORA_WEIGHT \
  --pretrained_model_path $PRETRAINED_MODEL_PATH \
  --finetuned_model_path $FINETUNED_MODEL_PATH \
  --test_image_path $TEST_IMAGE_PATH \
  --save_fig_path $SAVE_FIG_PATH \
  --num_images $NUM_IMAGES \
  --num_inference_steps $NUM_INFERENCE_STEPS \
  --guidance_scale $GUIDANCE_SCALE \
  --seed $SEED 
  



