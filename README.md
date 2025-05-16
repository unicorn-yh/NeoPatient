<div align="center">   <img src="figure/NeoPatient.png" width="400em" />   <p align="center"><strong>🩺 NeoPatient: Generate Lifelike Virtual Patients with Multimodal Medical Generative Models</strong></p> </div>

------



<div align="center" style="line-height: 1;">
  <a href="https://github.com/unicorn-yh/NeoPatient/blob/main/LICENSE">
    <img alt="MIT License"
      src="https://img.shields.io/badge/Code%20License-MIT-brightgreen?logo=open-source-initiative&logoColor=white"/>
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch"
      src="https://img.shields.io/badge/PyTorch-2.0+-%23EE4C2C?logo=pytorch&logoColor=white"/>
  </a>
  <a href="https://huggingface.co/stabilityai/stable-diffusion-2-1">
    <img alt="Hugging Face"
      src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Stability%20AI-ffc107?color=ffc107&logoColor=white"/>
   </a>
</div>



## 🧬 Overview

**NeoPatient** is a powerful framework for generating synthetic virtual patient data using *text-to-image diffusion models* fine-tuned on radiology reports. It bridges the gap between textual medical descriptions and realistic imaging data to accelerate data availability for medical AI research.

> 💡 Fine-tune **Stable Diffusion v2.1** on the ROCO dataset using **LoRA**, and generate high-quality, medically coherent images from clinical prompts.

------

## 📦 Dataset

- Download the preprocessed **ROCO** dataset from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/c10a6bfc0fb74fd28cbd/)

- Place it in `./dataset/`

- Structure:

  ```
  ./dataset/
  ├── train/
  │   └── images/
  └── metadata.jsonl
  ```

- Dataset size:

  - Train: 65,420 images
  - Validation: 8,172 images
  - Test: 8,176 images

- Each line in `metadata.jsonl` contains:

  - `"file_name"`: path to image
  - `"text"`: corresponding medical caption

------

## 🚀 Quickstart

### 1. Setup

```bash
git clone https://github.com/unicorn-yh/NeoPatient
cd NeoPatient
pip install -r requirements.txt
```

For faster downloads in China:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Download Pretrained Model

Get **Stable Diffusion v2.1**:

- From [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- Or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e2be80c926464046a661/)

------

## 🎯 Fine-Tuning with LoRA

Modify training hyperparameters in `train/run.sh`:

```bash
bash run.sh
```

Key flags:

- `LORA_RANK`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`
- `VALIDATION_PROMPT`: example prompt for monitoring generation quality

Train using single GPU with:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Multi-GPU with `accelerate`

```bash
accelerate launch --multi_gpu python lora_train.py \
  --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
  ...
```

More: [Accelerate Docs](https://huggingface.co/docs/accelerate/basic_tutorials/launch)

------

## 🖼 Inference & Evaluation

After training:

```bash
cd inference
bash run_test.sh
```

Evaluate with FID, CLIP Score, IS, LPIPS:

```bash
cd evaluation
bash run_eval.sh
```

------

## ⚙️ Hardware Tips

- Minimum: 1 GPU with 24GB memory (`gradient_checkpointing` + `fp16`)
- Optimal: GPUs with ≥30GB for faster training and larger batch sizes

------

## 📚 References

1. Rombach et al. (CVPR 2022). *High-resolution image synthesis with latent diffusion models*
2. von Platen et al. (Hugging Face). *Diffusers: State-of-the-art diffusion models*

------

## 🧠 Why Use NeoPatient?

- 🔬 Enables multimodal medical data generation
- 🏥 Empowers low-resource settings with synthetic EHRs
- 🤖 Boosts training data for AI in radiology, safely and ethically

------

