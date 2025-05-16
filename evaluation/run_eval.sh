#!/bin/bash

# Arguments for the evaluation script
MODEL="diffusion"
RANK=8
DATA=20

# Path configurations
OUTPUT_PATH="./save_score/$MODEL/rank${RANK}_data${DATA}"
IMAGE_FOLDERS_PATH="../inference/figure/$MODEL/rank${RANK}_data${DATA}"
TEST_IMAGE_PATH="../dataset/test" 

# Boolean flags to control which scores are calculated
CALCULATE_CLIP_SCORE=True    # True or False
CALCULATE_FID_SCORE=True     # True or False
CALCULATE_INCEPTION_SCORE=True # True or False
CALCULATE_LPIPS_SCORE=True   # True or False

# Optional arguments for specific scores (using defaults from eval.py)
IS_BATCH_SIZE=32
IS_N_SPLITS=10
LPIPS_NET='alex' # 'alex' or 'vgg'

# Common arguments for image identification (using defaults from eval.py)
GENERATED_IMAGE_FILENAME="fig_1.png"
ID_EXTENSION_TO_REPLACE=".jpg"

# --- Construct the python command ---
PYTHON_COMMAND="python eval.py \
  --output_path \"$OUTPUT_PATH\" \
  --image_folders_path \"$IMAGE_FOLDERS_PATH\" \
  --test_image_path \"$TEST_IMAGE_PATH\" \
  --is_batch_size $IS_BATCH_SIZE \
  --is_n_splits $IS_N_SPLITS \
  --lpips_net \"$LPIPS_NET\" \
  --generated_image_filename \"$GENERATED_IMAGE_FILENAME\" \
  --id_extension_to_replace \"$ID_EXTENSION_TO_REPLACE\""

# Add boolean flags conditionally
if [ "$CALCULATE_CLIP_SCORE" = True ]; then
  PYTHON_COMMAND="$PYTHON_COMMAND --clip_score"
fi

if [ "$CALCULATE_FID_SCORE" = True ]; then
  PYTHON_COMMAND="$PYTHON_COMMAND --fid_score"
fi

if [ "$CALCULATE_INCEPTION_SCORE" = True ]; then
  PYTHON_COMMAND="$PYTHON_COMMAND --inception_score"
fi

if [ "$CALCULATE_LPIPS_SCORE" = True ]; then
  PYTHON_COMMAND="$PYTHON_COMMAND --lpips_score" # Corrected from LPIPS_score
fi

# --- Execute the evaluation script ---
echo "Executing command:"
echo "$PYTHON_COMMAND"
echo # Newline for better readability

# Ensure the output directory for scores exists (optional, as Python script also does this)
# mkdir -p "$(dirname "$OUTPUT_PATH")" # dirname might be tricky if OUTPUT_PATH is just a filename base

eval "$PYTHON_COMMAND"

echo "Evaluation script finished."
