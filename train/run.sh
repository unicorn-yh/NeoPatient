#!/bin/bash

# Arguments for the fine-tuning script
LORA_RANK=8  # Rank for LoRA configuration
DATA_SIZE=100 # Training data size
PRETRAINED_MODEL_PATH="/disks/disk5/private/liyonghui/stable-diffusion-2-1" # Path to the pretrained model
TRAIN_DATA_DIR="../dataset/train" # Path to the dataset directory
OUTPUT_DIR="../output/finetuned_model_rank$LORA_RANK" # Directory to save the fine-tuned model
LOG_DIR="../log/train_rank${LORA_RANK}_data${DATA_SIZE}.log"
VALIDATION_DIR="../test/figure/model_rank$LORA_RANK/datasize_$DATA_SIZE"
IMAGE_COLUMN="image" # Column name for image filenames in metadata
CAPTION_COLUMN="text" # Column name for captions in metadata
BATCH_SIZE=1 # Training batch size
NUM_EPOCHS=20 # Number of training epochs
LEARNING_RATE=1e-4 # Learning rate for the optimizer
LR_SCHEDULER="constant" # Type of learning rate scheduler
LR_WARMUP_STEPS=0 # Warmup steps for the learning rate
RESOLUTION=512 # Resolution for the input images
SEED=42 # Random seed for reproducibility
VALIDATION_PROMPT="Abdominal CT scan shows a dilated appendix measuring 9mm in diameter, with surrounding fat stranding indicative of acute appendicitis."
MIXED_PRECISION="fp16"
NUM_VALIDATION_IMAGES=1
CHECKPOINTING_STEPS=1000

# Execute the fine-tuning script
python lora_train.py \
  --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
  --train_data_dir $TRAIN_DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --image_column $IMAGE_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --train_batch_size $BATCH_SIZE \
  --max_train_samples $DATA_SIZE \
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


