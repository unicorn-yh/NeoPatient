#!/bin/bash

# Arguments for the fine-tuning script
LORA_RANK=128  # Rank for LoRA configuration
PRETRAINED_MODEL_PATH="../stable-diffusion-2-1" # Path to the pretrained model
TRAIN_DATA_DIR="../dataset/nobel" # Path to the dataset directory
OUTPUT_DIR="../output/fine_tuned_model_$LORA_RANK" # Directory to save the fine-tuned model
LOG_DIR="../log/train_$LORA_RANK.log"
VALIDATION_DIR="../test/figure/nobel_$LORA_RANK"
IMAGE_COLUMN="image" # Column name for image filenames in metadata
CAPTION_COLUMN="text" # Column name for captions in metadata
BATCH_SIZE=1 # Training batch size
NUM_EPOCHS=60 # Number of training epochs
LEARNING_RATE=1e-4 # Learning rate for the optimizer
LR_SCHEDULER="constant" # Type of learning rate scheduler
LR_WARMUP_STEPS=0 # Warmup steps for the learning rate
RESOLUTION=512 # Resolution for the input images
SEED=42 # Random seed for reproducibility
VALIDATION_PROMPT="Jack Ma wearing a baseball cap, with a serious expression, and a simple necklace, in the style of Nobel Laureate."
MIXED_PRECISION="fp16"
NUM_VALIDATION_IMAGES=1
CHECKPOINTING_STEPS=5000

# Execute the fine-tuning script
python lora_train.py \
  --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
  --train_data_dir $TRAIN_DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --image_column $IMAGE_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --train_batch_size $BATCH_SIZE \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler $LR_SCHEDULER \
  --lr_warmup_steps $LR_WARMUP_STEPS \
  --checkpointing_steps $CHECKPOINTING_STEPS \
  --mixed_precision $MIXED_PRECISION \
  --resolution $RESOLUTION \
  --rank $LORA_RANK \
  --seed $SEED \
  --log_dir $LOG_DIR \
  --num_validation_images $NUM_VALIDATION_IMAGES \
  --validation_prompt "$VALIDATION_PROMPT" \
  --validation_dir $VALIDATION_DIR 


