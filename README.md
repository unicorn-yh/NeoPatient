<div align="center">
  <img src="figure/NeoPatient.png" width="400em" ></img>
  <p align="center">
    <strong>Multimodal EHR Synthesis for Virtual Patients Using Medical Text-to-Image Generation</strong>
  </p>
</div>




The `lora_train.py` script shows how to fine-tune stable diffusion model using LoRA on your own dataset.

<br>

## Running locally with PyTorch

Download **Stable Diffusion v2.1** from Hugging Face: **[Stable-Diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)**

Before running the scripts, make sure to install the library's training dependencies:

```bash
git clone https://github.com/unicorn-yh/Medical-Image-Generation
cd train
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

<br>

### Hardware

With `gradient_checkpointing` and `mixed_precision` it should be possible to fine tune the model on a single 24GB GPU. For higher `batch_size` and faster training it's better to use GPUs with >30GB memory.

***Note: Change the hyperparameters in the*** `run.sh` ***based on your own requirements.***

```sh
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
VALIDATION_PROMPT="Abdominal CT scan shows a dilated appendix measuring 9mm in diameter, with surrounding fat stranding indicative of acute appendicitis."
MIXED_PRECISION="fp16"
NUM_VALIDATION_IMAGES=1
CHECKPOINTING_STEPS=5000
```



Once the training is finished the model will be saved in the `output_dir` specified in the command. 

<br>

###  Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. Follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch) for running distributed training with `accelerate`. Here is an example command in `run.sh`:

```sh
# Execute the fine-tuning script
accelerate launch --multi_gpu python lora_train.py \
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
```

<br>

## Dataset

The dataset was organized in [`./dataset`](https://github.com/unicorn-yh/NeoPatient/tree/main/dataset), where the [`train/images`](https://github.com/unicorn-yh/NeoPatient/tree/main/dataset/train/images) folder in the path contains the training  images. The structure ensures compatibility with training pipelines. Each entry in `metadata.jsonl` contains the file path of the image and its corresponding descriptive caption.


TRAIN DATASIZE = 65420

TEST DATASIZE = 8176

VALIDATION DATASIZE = 8172



#### References:

[1] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 10684–10695).

[2] von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., Nair, D., Paul, S., Berman, W., Xu, Y., Liu, S., & Wolf, T. (2022). *Diffusers: State-of-the-art diffusion models* [GitHub repository]. GitHub. https://github.com/huggingface/diffusers



