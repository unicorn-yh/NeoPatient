from diffusers import StableDiffusionPipeline
import os
import argparse
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # set your own available cuda devices
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print("Torch version:",torch.__version__)
print("Cuda version:",torch.version.cuda)
print("Cuda name:",torch.cuda.get_device_name(0))

def get_image_caption(test_metadata_json_path):
    image_data = {}
    with open(test_metadata_json_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            image_name = data["file_name"]
            image_name = image_name.split("/")[-1]
            image_data[image_name] = data["text"]
    return image_data

def generate_test_images(args, test_image_ls):
    image_data = get_image_caption(os.path.join(args.test_image_path,"metadata.jsonl"))
    for test_image in test_image_ls:
        prompt = image_data[test_image]
        # prompt = ' '.join(prompt.split()[:3])
        # prompt = prompt.lower().replace(" ","_")
        # prompt = prompt.replace(":","").replace(".","")
        
        pipe = StableDiffusionPipeline.from_pretrained(   # Load the pipeline
            args.pretrained_model_path, torch_dtype=torch.float16
        )
        if args.use_lora_weight:
            pipe.load_lora_weights(args.finetuned_model_path)
            output_image_dir = (os.path.join(args.save_fig_path,test_image.split(".")[0]))
        else:
            output_image_dir = (os.path.join('/'.join(args.save_fig_path.split("/")[:-1]),"base",test_image.split(".")[0]))
        pipe.to("cuda")
        os.makedirs(output_image_dir, exist_ok=True) # Ensure output directory exists
        generator = torch.manual_seed(args.seed)  # Generate images
        for i in range(args.num_images):
            image = pipe(prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
            output_path = os.path.join(output_image_dir, f"fig_{i+1}.png")
            image.save(output_path)
            print(f"Saved image to {output_path}")

    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_path",type=str,default="../stable-diffusion-2-1",required=True,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--finetuned_model_path",type=str,default="../output/diffusion/modelckpt_rank8_data20/pytorch_lora_weights.safetensors",required=True,help="Path to finetuned model.",)
    parser.add_argument("--test_image_path",type=str,default="../dataset/test",required=True,help="Path of the test image dataset.",)
    parser.add_argument("--save_fig_path",type=str,default="./figure/diffusion/rank8_data20",required=True,help="Path to save the generated inference image.",)
    parser.add_argument("--num_images",type=int,default=1,required=True,help="Number of images to generate.",)
    parser.add_argument("--use_lora_weight",type=bool,default=True,required=True,help="Use LoRA weight for finetuned model inference, and do not use for pretrained model inference.",)
    parser.add_argument("--num_inference_steps", type=int, default=30, required=True, help="Total denoising steps during inference.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, required=True, help="Controls prompt adherence, higher means more guidance.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inference.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pick_images = [
        "ROCO_00084.jpg", "ROCO_00138.jpg", "ROCO_00264.jpg", "ROCO_00350.jpg",
        "ROCO_00454.jpg", "ROCO_00509.jpg", "ROCO_00599.jpg", "ROCO_00611.jpg",
        "ROCO_00723.jpg", "ROCO_00747.jpg", "ROCO_02754.jpg", "ROCO_03231.jpg"
    ]
    generate_test_images(args=args, test_image_ls=pick_images)
    
if __name__ == "__main__":
    main()

